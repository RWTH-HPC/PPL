package de.parallelpatterndsl.patterndsl.performance;

import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.mapping.Mapping;
import de.parallelpatterndsl.patterndsl.mapping.StepMapping;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplit;
import de.parallelpatterndsl.patterndsl.patternSplits.PatternSplit;
import de.parallelpatterndsl.patterndsl.teams.Team;

import java.util.Set;

public abstract class PerformanceModel {

    protected ExecutionCostsEstimator executionCostsEstimator;

    protected NetworkCostsEstimator networkCostsEstimator;

    protected Network network;

    public PerformanceModel(ExecutionCostsEstimator executionCostsEstimator, NetworkCostsEstimator networkCostsEstimator, Network network) {
        this.executionCostsEstimator = executionCostsEstimator;
        this.networkCostsEstimator = networkCostsEstimator;
        this.network = network;
    }

    public abstract double evaluate(StepMapping stepMapping, Mapping mapping);

    public ExecutionCostsEstimator getExecutionCostsEstimator() {
        return this.executionCostsEstimator;
    }

    public NetworkCostsEstimator getNetworkCostsEstimator() {
        return this.networkCostsEstimator;
    }


}
