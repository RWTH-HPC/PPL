package de.parallelpatterndsl.patterndsl.performance.simple;

import com.google.common.collect.Sets;
import com.google.common.collect.Table;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplit;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;
import de.parallelpatterndsl.patterndsl.mapping.Mapping;
import de.parallelpatterndsl.patterndsl.mapping.StepMapping;
import de.parallelpatterndsl.patterndsl.patternSplits.PatternSplit;
import de.parallelpatterndsl.patterndsl.performance.PerformanceModel;
import de.parallelpatterndsl.patterndsl.teams.Team;
import de.parallelpatterndsl.patterndsl.teams.Teams;

import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Imlementation of a performance model based on the roofline model.
 * Details can be found in the thesis "Global Optimization of Parallel Pattern-based Algorithms for Heterogeneous Architectures" in git.
 */
public class SimplePerformanceModel extends PerformanceModel {

    private final double overlap;

    public SimplePerformanceModel(Network network, double overlap) {
        super(new SimpleExecutionCostsEstimator(), new SimpleNetworkCostsEstimator(), network);
        this.overlap = overlap;
    }

    @Override
    public double evaluate(StepMapping stepMapping, Mapping mapping) {
        double maxCosts = 0.0;
        for (Team team : stepMapping.teams()) {
            Set<PatternSplit> patternSplits = stepMapping.get(team);
            Set<DataSplit> dataSplits = patternSplits.stream().flatMap(j -> j.getInputDataSplits().stream()).collect(Collectors.toSet());

            double executionCosts = super.executionCostsEstimator.estimateAll(patternSplits, team);
            double networkCosts = super.networkCostsEstimator.estimateAll(dataSplits, team, stepMapping.teams(), mapping, super.network, true);
            double costs = (1 - Double.min(networkCosts / executionCosts, overlap)) * executionCosts + networkCosts;

            if (costs > maxCosts) {
                maxCosts = costs;
            }
        }

        return maxCosts;
    }

}
