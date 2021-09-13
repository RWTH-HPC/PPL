package de.parallelpatterndsl.patterndsl.performance;

import de.parallelpatterndsl.patterndsl.dataSplits.DataSplit;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.mapping.Mapping;
import de.parallelpatterndsl.patterndsl.teams.Team;

import java.util.Set;

public interface NetworkCostsEstimator {

    double LATENCY_TO_SECONDS = 1e9; // Nanoseconds = 1e9.

    double BANDWIDTH_TO_SECONDS = 1e6; // Megabyte/s = 1e6.

    /**
     *
     * @param split
     * @param team
     * @param history
     * @return
     */
    double estimate(DataSplit split, Team team, Set<Team> teams, Mapping history, Network network, boolean applyPenalty);

    /**
     *
     * @param splits
     * @param team
     * @param history
     * @return
     */
    double estimateAll(Set<DataSplit> splits, Team team, Set<Team> teams, Mapping history, Network network, boolean applyPenalty);

    /**
     *
     * @param from
     * @param to
     * @return
     */
    double latencyPenalty(Team from, Team to, Network network);

}
