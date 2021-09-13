package de.parallelpatterndsl.patterndsl.mapping;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplit;
import de.parallelpatterndsl.patterndsl.patternSplits.FusedPatternSplit;
import de.parallelpatterndsl.patterndsl.patternSplits.PatternSplit;
import de.parallelpatterndsl.patterndsl.teams.Team;
import de.parallelpatterndsl.patterndsl.teams.Teams;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Mapping hypothesis up to some time step.
 */
public class Mapping implements Cloneable {

    private final LinkedList<StepMapping> stepMappingStack;

    private final HashMap<DataSplit, HashSet<Team>> dataSplitHolderLookupTable;

    public Mapping() {
        this.stepMappingStack = new LinkedList<>();
        this.dataSplitHolderLookupTable = new HashMap<>();
    }

    public void push(StepMapping stepMapping) {
        this.stepMappingStack.addLast(stepMapping);
        this.updateLookupTable(stepMapping);
    }

    public StepMapping current() {
        return this.stepMappingStack.getLast();
    }

    public StepMapping assignmentOf(int step) {
        return stepMappingStack.get(step);
    }

    public int currentStep() {
        return this.stepMappingStack.size();
    }

    public Mapping clone() {
        Mapping cloned = new Mapping();
        for (StepMapping stepMapping : this.stepMappingStack) {
            cloned.push(stepMapping);
        }
        return cloned;
    }

    private void updateLookupTable(StepMapping stepMapping) {
        // TODO: Pruning to avoid tables from growing too large.

        for (PatternSplit split : stepMapping.splits()) {
            for (DataSplit dataSplit : split.getOutputDataSplits()) {
                HashSet<Team> teams = new HashSet<>();
                teams.add(stepMapping.get(split));
                this.dataSplitHolderLookupTable.put(dataSplit, teams);
            }
        }

        /*
        // Either overwrite or make list and append.
        for (PatternSplit split : stepMapping.splits()) {
            for (DataSplit dataSplit : split.getInputDataSplits()) {
                if (this.dataSplitHolderLookupTable.containsKey(dataSplit)) {
                    HashSet<Team> teams = this.dataSplitHolderLookupTable.get(dataSplit);
                    teams.add(stepMapping.get(split));
                } else {
                    HashSet<Team> teams = new HashSet<>();
                    teams.add(stepMapping.get(split));
                    this.dataSplitHolderLookupTable.put(dataSplit, teams);
                }
            }
        }*/
    }

    public LinkedList<StepMapping> getStepMappingStack() {
        return stepMappingStack;
    }

    /**
     * For a given set of data splits, the method estimates those teams, which has processed the data splits previously.
     * If multiple teams have processed a data split simultaneously, the team with lowest distance is chosen.
     * @param dataSplits
     * @param teams
     * @param network
     * @return
     */
    public Table<DataSplit, Team, Team> dataSplitHolder(Set<DataSplit> dataSplits, Set<Team> teams, Network network) {
        Table<DataSplit, Team, Team> lastAccessorTable = HashBasedTable.create();

        for (DataSplit split : dataSplits) {
            if (this.dataSplitHolderLookupTable.containsKey(split)) {
                HashSet<Team> fromTeams = this.dataSplitHolderLookupTable.get(split);
                for (Team toTeam : teams) {
                    Team closestTeam = null;
                    Teams.TeamDistance closestDistance = Teams.TeamDistance.NETWORK;
                    for (Team fromTeam : fromTeams) {
                        Teams.TeamDistance distance = Teams.distance(fromTeam, toTeam, network);
                        if (closestTeam == null || distance.ordinal() < closestDistance.ordinal()) {
                            closestTeam = fromTeam;
                            closestDistance = distance;
                        }
                    }
                    lastAccessorTable.put(split, toTeam, closestTeam);
                }
            }
        }

        return lastAccessorTable;
    }

    public void toJSONFile(String path, String algorithm, String cluster, double score) throws IOException {
        JsonObject mainObject = new JsonObject();
        mainObject.addProperty("Algorithm", algorithm);
        mainObject.addProperty("Cluster", cluster);
        mainObject.addProperty("Score", score);
        mainObject.addProperty("Timestamp", new java.util.Date().toString());

        JsonArray stepArray = new JsonArray();
        for (StepMapping stepMapping : this.stepMappingStack) {
            JsonObject stepObject = new JsonObject();
            stepObject.addProperty("Step", stepMapping.getStep());

            JsonArray nodesObject = new JsonArray();
            Map<PatternNode, List<PatternSplit>> groupBy = stepMapping.splits()
                    .stream()
                    .filter(split -> !(split instanceof FusedPatternSplit))
                    .collect(Collectors.groupingBy(PatternSplit::getNode));
            for (PatternNode node : groupBy.keySet()) {
                JsonObject nodeObject = new JsonObject();

                if (node instanceof ParallelCallNode) {
                    nodeObject.addProperty("Node", ((ParallelCallNode) node).getFunctionIdentifier());
                }

                JsonArray assignmentObject = new JsonArray();
                for (PatternSplit split : groupBy.get(node).stream().sorted(Comparator.comparingInt(s -> s.getStartIndices()[0])).collect(Collectors.toCollection(LinkedHashSet::new))) {
                    Team team = stepMapping.get(split);

                    JsonObject splitObject = new JsonObject();
                    JsonArray indicesObject = new JsonArray();
                    for (int index : split.getStartIndices()) {
                        indicesObject.add(index);
                    }
                    JsonArray lengthObject = new JsonArray();
                    for (long length : split.getLengths()) {
                        lengthObject.add(length);
                    }

                    splitObject.add("Indices", indicesObject);
                    splitObject.add("Lengths", lengthObject);
                    splitObject.addProperty("Node", team.getDevice().getParent().getIdentifier());
                    splitObject.addProperty("Device", team.getDevice().getIdentifier());
                    splitObject.addProperty("Processor", team.getProcessor().getIdentifier());
                    splitObject.addProperty("Cores", team.getCores());

                    assignmentObject.add(splitObject);
                }

                nodeObject.add("Assignment", assignmentObject);
                nodesObject.add(nodeObject);
            }

            for (FusedPatternSplit pipe : stepMapping.splits()
                    .stream()
                    .filter(split -> split instanceof FusedPatternSplit)
                    .map(split -> (FusedPatternSplit) split)
                    .sorted(Comparator.comparingInt(s -> s.getStartIndices()[0]))
                    .collect(Collectors.toCollection(LinkedHashSet::new)))
            {
                JsonObject pipeNode = new JsonObject();
                Team team = stepMapping.get(pipe);
                pipeNode.addProperty("Node", team.getDevice().getParent().getIdentifier());
                pipeNode.addProperty("Device", team.getDevice().getIdentifier());
                pipeNode.addProperty("Processor", team.getProcessor().getIdentifier());
                pipeNode.addProperty("Cores", team.getCores());

                JsonArray pipeArray = new JsonArray();
                for (PatternSplit split : pipe.getJobs()) {
                    JsonObject splitObject = new JsonObject();
                    JsonArray indicesObject = new JsonArray();
                    for (int index : split.getStartIndices()) {
                        indicesObject.add(index);
                    }
                    JsonArray lengthObject = new JsonArray();
                    for (long length : split.getLengths()) {
                        lengthObject.add(length);
                    }

                    splitObject.addProperty("Node", ((ParallelCallNode) split.getNode()).getFunctionIdentifier());
                    splitObject.add("Indices", indicesObject);
                    splitObject.add("Lengths", lengthObject);
                    pipeArray.add(splitObject);
                }
                pipeNode.add("Fusion", pipeArray);
                nodesObject.add(pipeNode);
            }

            stepObject.add("Nodes", nodesObject);
            stepArray.add(stepObject);
        }
        mainObject.add("stepMappings", stepArray);

        FileWriter writer = new FileWriter(path);
        writer.write(mainObject.toString());
        writer.flush();
        writer.close();
    }

}
