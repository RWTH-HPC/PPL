package de.parallelpatterndsl.patterndsl.performance;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.patternSplits.PatternSplit;
import de.parallelpatterndsl.patterndsl.teams.Team;

import java.util.Set;

public interface ExecutionCostsEstimator extends APTVisitor {

    double FREQUENCY_TO_SECONDS = 1e6; // Megahertz = 1e6.

    /**
     *
     * @param split
     * @param team
     * @return
     */
     double estimate(PatternSplit split, Team team);

    /**
     *
     */
    double estimateAll(Set<PatternSplit> splits, Team team);

}
