package de.parallelpatterndsl.patterndsl.mapping;

import de.parallelpatterndsl.patterndsl.patternSplits.PatternSplit;
import de.parallelpatterndsl.patterndsl.teams.Team;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * The StepMapping class represents the assignment of a single step.
 */
public class StepMapping {

    private final int step;

    private final HashMap<PatternSplit, Team> assignment;

    public StepMapping(int step) {
        this.step = step;
        this.assignment = new HashMap<>();
    }

    public void assign(PatternSplit patternSplit, Team team) {
        this.assignment.put(patternSplit, team);
    }

    public void free(PatternSplit patternSplit) {
        this.assignment.remove(patternSplit);
    }

    public Set<Team> teams() {
        return new HashSet<>(this.assignment.values());
    }

    public Set<PatternSplit> splits() {
        return this.assignment.keySet();
    }

    public Team get(PatternSplit patternSplit) {
        return this.assignment.getOrDefault(patternSplit, null);
    }

    public Set<PatternSplit> get(Team team) {
        return this.assignment.entrySet().stream()
                .filter(e -> e.getValue().equals(team))
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet());
    }

    public int getStep() {
        return this.step;
    }

}
