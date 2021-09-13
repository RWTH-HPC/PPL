package de.parallelpatterndsl.patterndsl;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Sets;
import com.google.common.collect.Table;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.MapNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.ReduceNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;
import de.parallelpatterndsl.patterndsl.performance.PerformanceModel;
import de.parallelpatterndsl.patterndsl.performance.simple.SimplePerformanceModel;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.mapping.Mapping;
import de.parallelpatterndsl.patterndsl.mapping.StepMapping;
import de.parallelpatterndsl.patterndsl.patternSplits.FusedPatternSplit;
import de.parallelpatterndsl.patterndsl.patternSplits.IOPatternSplit;
import de.parallelpatterndsl.patterndsl.patternSplits.PatternSplit;
import de.parallelpatterndsl.patterndsl.patternSplits.ParallelPatternSplit;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplit;
import de.parallelpatterndsl.patterndsl.teams.Team;
import de.parallelpatterndsl.patterndsl.teams.Teams;
import gurobi.*;
import org.antlr.v4.runtime.misc.Pair;
import org.antlr.v4.runtime.misc.Triple;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Generates the mapping by optimization the Inter-Node Dataflow Efficiency.
 */
public class MappingGenerator {

    private static final int MAX_MOVES = 1000;
    
    private final FlatAPT flatAPT;

    private final Network network;

    private final PerformanceModel model;

    private final int lookahead;

    private GRBEnv grbEnv;

    /**
     * Constructs a MappingGenerator object.
     * @param flatAPT - algorithm representation.
     * @param network - cluster description.
     * @param model - performance model to be optimized.
     * @param lookahead - lookahead to be used.
     */
    public MappingGenerator(FlatAPT flatAPT, Network network, PerformanceModel model, int lookahead) {
        this.flatAPT = flatAPT;
        this.network = network;
        this.model = model;
        this.lookahead = lookahead;

        try {
            this.grbEnv = new GRBEnv(true);
            this.grbEnv.set(GRB.IntParam.Method, 4);
            this.grbEnv.set(GRB.DoubleParam.TimeLimit, 240.0);

            this.grbEnv.start();
        } catch (GRBException e) {
            e.printStackTrace();
        }
    }

    /**
     * Constructs a MappingGenerator object.
     * @param flatAPT - algorithm representation.
     * @param network - cluster description.
     * @param model - performance model to be optimized.
     * @param lookahead - lookahead to be used.
     */
    public MappingGenerator(FlatAPT flatAPT, Network network, PerformanceModel model, int lookahead, int timeLimit) {
        this.flatAPT = flatAPT;
        this.network = network;
        this.model = model;
        this.lookahead = lookahead;

        try {
            this.grbEnv = new GRBEnv(true);
            this.grbEnv.set(GRB.IntParam.Method, 4);
            this.grbEnv.set(GRB.DoubleParam.TimeLimit, timeLimit);

            this.grbEnv.start();
        } catch (GRBException e) {
            e.printStackTrace();
        }
    }

    /**
     * Estimates the optimal mapping with respect to the provided performance model
     * @return Mapping object.
     */
    public Pair<Mapping, Double> generate() {
        Mapping mapping = new Mapping();
        double score = 0.0;

        Set<Team> firstTeams = Teams.initialTeams(network);
        Triple<StepMapping, Double, Double> firstStepMapping = stairClimbing(mapping, firstTeams, Integer.min(lookahead, flatAPT.size() - 1));
        mapping.push(firstStepMapping.a);
        score += firstStepMapping.b;

        for (int step = 1; step < this.flatAPT.size(); step++) {
            Set<Team> initialTeams = mapping.current().teams().stream().map(Team::new).collect(Collectors.toSet());

            Triple<StepMapping, Double, Double> stepMapping = stairClimbing(mapping, initialTeams, Integer.min(lookahead, flatAPT.size() - 1 - step));
            mapping.push(stepMapping.a);
            score += stepMapping.b;
        }

        System.out.println("Estimated mapping with score: " + score);
        Mapping fusedMapping = fuse(mapping);
        return new Pair<Mapping, Double>(fusedMapping, score);
    }

    /**
     * Direct implementation of the stair climbing algorithm. More information: wiki or Thesis.
     * @param mapping - the mapping hypothesis up to the current step.
     * @param initialTeams - the initial teams for the search.
     * @param lookahead - the lookahead to be used.
     * @return (StepMapping, score, lookaheadScore)
     */
    private Triple<StepMapping, Double, Double> stairClimbing(Mapping mapping, Set<Team> initialTeams, int lookahead) {
        // 1. Generate initial assignment.
        StepMapping stepMapping = assignStep(mapping, initialTeams);
        double bestCosts = this.model.evaluate(stepMapping, mapping);

        // 2. Lookahead.
        double bestCostsLookahead = bestCosts;
        if (lookahead > 0) {
            Mapping lookaheadMapping = mapping.clone();
            lookaheadMapping.push(stepMapping);

            // Generate new java objects for teams. Otherwise scale move changes cores.
            Set<Team> lookaheadTeams = stepMapping.teams().stream().map(Team::new).collect(Collectors.toSet());
            bestCostsLookahead += stairClimbing(lookaheadMapping, lookaheadTeams, lookahead - 1).c;
        }

        // 3. Search.

        // Collapse hypotheses to enforce plateaus.
        LinkedList<Set<Team>> movesQueue = move(stepMapping.teams());
        boolean plateau = false;

        // Climbing.
        for (int i = 0; i < MAX_MOVES; i++) {
            StepMapping bestMoveStepMapping = null;
            double bestMoveCosts = Double.POSITIVE_INFINITY;
            double bestMoveCostsLookahead = Double.POSITIVE_INFINITY;

            for (Set<Team> move : movesQueue) {
                StepMapping moveStepMapping = assignStep(mapping, move);
                double moveCosts = this.model.evaluate(moveStepMapping, mapping);

                double moveCostsLookahead = moveCosts;
                if (lookahead > 0) {
                    Mapping lookaheadMapping = mapping.clone();
                    lookaheadMapping.push(moveStepMapping);

                    Set<Team> lookaheadTeams = moveStepMapping.teams().stream().map(Team::new).collect(Collectors.toSet());
                    moveCostsLookahead += stairClimbing(lookaheadMapping, lookaheadTeams, lookahead - 1).c;
                }

                if (moveCostsLookahead < bestMoveCostsLookahead) {
                    bestMoveCosts = moveCosts;
                    bestMoveCostsLookahead = moveCostsLookahead;
                    bestMoveStepMapping = moveStepMapping;
                }
            }

            if (bestMoveCostsLookahead < bestCostsLookahead) { // Move
                movesQueue.clear();
                movesQueue.addAll(move(bestMoveStepMapping.teams()));

                stepMapping = bestMoveStepMapping;
                bestCosts = bestMoveCosts;
                bestCostsLookahead = bestMoveCostsLookahead;

                plateau = false;
            }  else if (bestMoveCostsLookahead == bestCostsLookahead && !plateau) { // Jump
                movesQueue.clear();
                movesQueue.addAll(jump(bestMoveStepMapping.teams()));

                stepMapping = bestMoveStepMapping;
                bestCosts = bestMoveCosts;
                bestCostsLookahead = bestMoveCostsLookahead;

                plateau = true;
            } else { // Stop
                break;
            }
        }

        return new Triple<>(stepMapping, bestCosts, bestCostsLookahead);
    }

    /**
     * Implementation of the single-step mapping using Gurobi and a direct ILP formulation.
     * @param mapping - the mapping hypothesis up to the current step.
     * @param teams - the teams to be used.
     * @return StepMapping object
     */
    private StepMapping assignStep(Mapping mapping, Set<Team> teams)  {
        List<PatternSplit> patternSplits = this.flatAPT.getSplits(mapping.currentStep())
                .stream()
                .filter(j -> j instanceof ParallelPatternSplit || j instanceof IOPatternSplit)
                .collect(Collectors.toList());
        
        // Nothing to do: "Data-init-step"
        if (patternSplits.isEmpty()) {
            return new StepMapping(mapping.currentStep());
        }

        // Add CPU teams for IOJob.
        if (patternSplits.stream().anyMatch(j -> j instanceof IOPatternSplit) && teams.stream().noneMatch(t -> t.getDevice().getType().equals("CPU"))) {
            Optional<Team> gpu = teams.stream().filter(team -> team.getDevice().getType().equals("GPU")).findAny();
            if (gpu.isPresent()) {
                teams.add(Teams.host(gpu.get().getDevice(), network));
            } else {
                Device device = network.getNodes().get(0).getDevices().stream().filter(device1 -> device1.getType().equals("CPU")).findFirst().get();
                Processor proc = device.getProcessor().get(0);
                teams.add(new Team(device, proc, proc.getCores()));
            }
        }

        Triple<StepMapping, Double, Double> greedy = bounds(mapping, teams);
        try {
            // Group pattern splits by the shared data splits.
            Map<DataSplit, Set<PatternSplit>> sharedDataSplitMap = new HashMap<>();
            for (PatternSplit patternSplit : patternSplits) {
                for (DataSplit pkg : patternSplit.getInputDataSplits()) {
                    Set<PatternSplit> jobsSharingPackage;
                    if (sharedDataSplitMap.containsKey(pkg)) {
                        jobsSharingPackage = sharedDataSplitMap.get(pkg);
                    } else {
                        jobsSharingPackage = new HashSet<>();
                        sharedDataSplitMap.put(pkg, jobsSharingPackage);
                    }
                    jobsSharingPackage.add(patternSplit);
                }
            }
            Table<DataSplit, Team, Team> dataSplitHolders = mapping.dataSplitHolder(sharedDataSplitMap.keySet(), teams, network);

            // Create ILP model.
            GRBModel model = new GRBModel(this.grbEnv);
            model.set(GRB.IntParam.OutputFlag, 0);

            // Define objective: Minimize runtime.
            GRBVar runtime = model.addVar(0.0, greedy.c, 0.0, GRB.CONTINUOUS, "runtime");
            GRBLinExpr objective = new GRBLinExpr();
            objective.addTerm(1.0, runtime);
            model.setObjective(objective, GRB.MINIMIZE);

            // 1. Assignment constraints: Each pattern split is assigned to one team.
            Table<Team, PatternSplit, GRBVar> assignmentVars = HashBasedTable.create();
            for (PatternSplit patternSplit : patternSplits) {
                GRBLinExpr assignmentConstraint = new GRBLinExpr();
                for (Team team : teams) {
                    if (patternSplit instanceof IOPatternSplit && !team.getDevice().getType().equals("CPU")) {
                        continue;
                    }

                    GRBVar assignmentVar = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "A_" + patternSplit.toString() + "," + team.toString());
                    assignmentConstraint.addTerm(1.0, assignmentVar);
                    assignmentVars.put(team, patternSplit, assignmentVar);
                }
                model.addConstr(1.0, GRB.EQUAL, assignmentConstraint, "AssignmentConstraint_" + patternSplit.toString());
            }

            // 2. Runtime constraints: Costs of each team are bounded by runtime variable.
            for (Team team : assignmentVars.rowKeySet()) {
                GRBLinExpr runtimeConstraint = new GRBLinExpr();

                // Execution costs.
                for (PatternSplit patternSplit : assignmentVars.row(team).keySet()) {
                    GRBVar assignmentVar = assignmentVars.get(team, patternSplit);
                    double executionCosts = this.model.getExecutionCostsEstimator().estimate(patternSplit, team);
                    runtimeConstraint.addTerm(executionCosts, assignmentVar);
                }

                // Network costs.
                for (DataSplit pkg : sharedDataSplitMap.keySet()) {
                    Set<PatternSplit> jobsSharingNetworkPackage = sharedDataSplitMap.get(pkg);
                    double bandwidthCosts = this.model.getNetworkCostsEstimator().estimate(pkg, team, teams, mapping, network, false);

                    Sets.SetView<PatternSplit> intersect = Sets.intersection(jobsSharingNetworkPackage, assignmentVars.row(team).keySet());
                    GRBVar[] vars = new GRBVar[intersect.size()];
                    int j = 0;
                    for (PatternSplit patternSplit : intersect) {
                        vars[j] = assignmentVars.get(team, patternSplit);
                        j++;
                    }

                    GRBVar resVar = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "MaxVar_" + team.toString());
                    model.addGenConstrOr(resVar, vars, "Maxi_" + team.toString());
                    runtimeConstraint.addTerm(bandwidthCosts, resVar);
                }

                // Extension: Communication costs (latency).
                Map<Team, Set<DataSplit>> dataSplitHolderMap = dataSplitHolders.column(team).entrySet().stream()
                        .collect(Collectors.groupingBy(Map.Entry::getValue, Collectors.mapping(Map.Entry::getKey, Collectors.toSet())));
                for (Team last : dataSplitHolderMap.keySet()) {
                    double latencyCosts = this.model.getNetworkCostsEstimator().latencyPenalty(last, team, this.network);

                    Set<PatternSplit> js = dataSplitHolderMap.get(last).stream().flatMap(p -> sharedDataSplitMap.get(p).stream()).collect(Collectors.toSet());
                    GRBVar[] vars = new GRBVar[js.size()];
                    int j = 0;
                    for (PatternSplit patternSplit : assignmentVars.row(team).keySet()) {
                        if (js.contains(patternSplit)) {
                            vars[j] = assignmentVars.get(team, patternSplit);
                            j++;
                        }

                        if (j == vars.length) {
                            break;
                        }
                    }

                    GRBVar resVar = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "MaxVar_" + team.toString());
                    model.addGenConstrOr(resVar, vars, "Maxi_" + team.toString());
                    runtimeConstraint.addTerm(latencyCosts, resVar);
                }

                model.addConstr(runtime, GRB.GREATER_EQUAL, runtimeConstraint, "RuntimeConstraint_" + team.toString());
            }

            model.optimize();
            int statusCode = model.get(GRB.IntAttr.Status);
            if (statusCode == GRB.Status.OPTIMAL) {
                StepMapping stepMapping = new StepMapping(mapping.currentStep());
                for (Team team : assignmentVars.rowKeySet()) {
                    for (PatternSplit patternSplit : assignmentVars.row(team).keySet()) {
                        int weight = (int) Math.round(assignmentVars.get(team, patternSplit).get(GRB.DoubleAttr.X));

                        if (weight == 1) {
                            stepMapping.assign(patternSplit, team);
                        }
                    }
                }

                model.dispose();
                if (stepMapping.splits().size() != patternSplits.size()) {
                    return null;
                }
                return stepMapping;
            }
        } catch (GRBException e) {
            e.printStackTrace();
        }

        return greedy.a;
    }

    /**
     * Estimates the bounds of the ILP by assigning the nodes of this step to the teams greedily.
     * @param mapping - the state summarizing all preceding assignments.
     * @param teams - the team to be used.
     * @return (StepMapping, lowerBound, upperBound)
     */
    private Triple<StepMapping, Double, Double> bounds(Mapping mapping, Set<Team> teams) {
        // ILP minimizes according to 0.0 overlap.
        SimplePerformanceModel modelWithoutOverlap = new SimplePerformanceModel(network, 0.0);

        Set<PatternSplit> patternSplits = this.flatAPT.getSplits(mapping.currentStep()).stream().filter(j -> j instanceof ParallelPatternSplit || j instanceof IOPatternSplit).collect(Collectors.toSet());
        StepMapping greedyStepMapping = new StepMapping(mapping.currentStep());

        // Assign IOJobs arbitrarily.
        if (patternSplits.stream().anyMatch(t -> t instanceof IOPatternSplit)) {
            Team ioTeam = teams.stream().filter(t -> t.getDevice().getType().equals("CPU")).findAny().get();
            for (PatternSplit patternSplit : patternSplits.stream().filter(t -> t instanceof IOPatternSplit).collect(Collectors.toSet())) {
                greedyStepMapping.assign(patternSplit, ioTeam);
            }
        }
        for (PatternSplit patternSplit : patternSplits.stream().filter(t -> t instanceof ParallelPatternSplit).collect(Collectors.toSet())) {
            Team bestTeam = null;
            double bestUpperScore = Double.POSITIVE_INFINITY;
            for (Team team : teams) {
                greedyStepMapping.assign(patternSplit, team);
                double upperScore = modelWithoutOverlap.evaluate(greedyStepMapping, mapping);
                greedyStepMapping.free(patternSplit);

                if (upperScore < bestUpperScore) {
                    bestUpperScore = upperScore;
                    bestTeam = team;
                }
            }

            if (bestTeam == null) {
                return new Triple<>(null, 0.0, Double.POSITIVE_INFINITY);
            }

            greedyStepMapping.assign(patternSplit, bestTeam);
        }

        double upperBound = modelWithoutOverlap.evaluate(greedyStepMapping, mapping);
        return new Triple<>(greedyStepMapping, 0.0, upperBound);
    }

    private LinkedList<Set<Team>> move(Set<Team> currentTeams) {
        LinkedList<Set<Team>> teams = new LinkedList<>();

        Set<Team> scaledTeams = Teams.scaleUp(currentTeams, 24);
        Set<Team> spreadTeams = Teams.spread(currentTeams, true, network);
        teams.add(scaledTeams);
        if (spreadTeams != null) {
            teams.add(spreadTeams);
        }

        return teams;
    }

    private LinkedList<Set<Team>> jump(Set<Team> currentTeams) {
        LinkedList<Set<Team>> teams = new LinkedList<>();
        Set<Team> spreadTeams = Teams.spread(currentTeams, false, network);
        Set<Team> gpuTeams = Teams.offloading(currentTeams, 84, network);
        if (spreadTeams != null) {
            teams.add(spreadTeams);
        }
        if (gpuTeams != null) {
            teams.add(gpuTeams);
        }

        return teams;
    }

    /**
     * Fuse-first heuristic optimizing the Intra-Node Dataflow Efficiency.
     * @param mapping - generated mapping.
     * @return Mapping with FusedPatternSplits.
     */
    private Mapping fuse(Mapping mapping) {
        Mapping finalMapping = new Mapping();
        StepMapping firstStepMapping = mapping.assignmentOf(0);
        finalMapping.push(firstStepMapping);

        HashMap<Team, Set<PatternSplit>> active_ = new HashMap<>();
        HashMap<Team, Set<PatternSplit>> active = new HashMap<>();
        for (Team team : firstStepMapping.teams()) {
            active_.put(team, firstStepMapping.get(team).stream().filter(j -> j instanceof ParallelPatternSplit).collect(Collectors.toSet()));
        }

        for (int step = 1; step < flatAPT.size(); step++) {
            StepMapping stepMapping = mapping.assignmentOf(step);

            LinkedList<PatternSplit> patternSplits = flatAPT.getSplits(step)
                    .stream()
                    .filter(j -> j instanceof ParallelPatternSplit)
                    .filter(j -> AbstractPatternTree.getFunctionTable().get(((ParallelCallNode) j.getNode()).getFunctionIdentifier()) instanceof MapNode)
                    .collect(Collectors.toCollection(LinkedList::new));
            for (Team team : active_.keySet()) {
                for (PatternSplit activePipe : active_.get(team)) {
                    for (PatternSplit patternSplit : patternSplits) {
                        if (team.getProcessor().equals(stepMapping.get(patternSplit).getProcessor())
                                && activePipe.getOutputDataSplits().containsAll(patternSplit.getInputDataSplits())) {
                            FusedPatternSplit pipe;
                            if (activePipe instanceof FusedPatternSplit) {
                                pipe = (FusedPatternSplit) activePipe;
                            } else {
                                pipe = new FusedPatternSplit((ParallelPatternSplit) activePipe);
                                mapping.assignmentOf(step - 1).free(activePipe);
                                mapping.assignmentOf(step - 1).assign(pipe, team);
                            }

                            pipe.append((ParallelPatternSplit) patternSplit);
                            stepMapping.free(patternSplit);
                            patternSplits.remove(patternSplit);

                            Set<PatternSplit> fused = active.getOrDefault(team, new HashSet<>());
                            fused.add(pipe);
                            active.put(team, fused);

                            break;
                        }
                    }
                }
            }
            for (Team team : stepMapping.teams()) {
                active.put(team, stepMapping.get(team));
            }

            active_.clear();
            active_.putAll(active);
            active.clear();

            if (!stepMapping.splits().isEmpty()) {
                finalMapping.push(stepMapping);
            }
        }

        return finalMapping;
    }

}
