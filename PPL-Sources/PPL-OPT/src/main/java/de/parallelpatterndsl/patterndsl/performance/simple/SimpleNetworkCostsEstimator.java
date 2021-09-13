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
import de.parallelpatterndsl.patterndsl.performance.NetworkCostsEstimator;
import de.parallelpatterndsl.patterndsl.teams.Team;
import de.parallelpatterndsl.patterndsl.teams.Teams;

import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class SimpleNetworkCostsEstimator implements NetworkCostsEstimator {

    @Override
    public double estimate(DataSplit split, Team team, Set<Team> teams, Mapping history, Network network, boolean applyPenalty) {
        Collection<Team> dataSplitHolders = history.dataSplitHolder(Sets.newHashSet(split), Sets.newHashSet(team), network).values();

        if (dataSplitHolders.isEmpty()) {
            double streams = Double.min(team.getDevice().getMaxMainBandwidth() / team.getDevice().getMainBandwidth(), team.getCores());
            double teamStreams = streams * (team.getCores() / (double) team.getCores());
            double networkCosts = split.getBytes() / (BANDWIDTH_TO_SECONDS * team.getDevice().getMainBandwidth() * teamStreams);
            if (applyPenalty) {
                networkCosts += team.getDevice().getMainLatency() / LATENCY_TO_SECONDS;
            }
            return networkCosts;
        }

        double networkCosts;
        Team fromTeam = dataSplitHolders.iterator().next();
        Teams.TeamDistance distance = Teams.distance(fromTeam, team, network);
        if (distance == Teams.TeamDistance.PROCESSOR) {
            Processor processor = fromTeam.getProcessor();
            int cacheLevel = processor.getCacheBandwidth().size() - 1;
            networkCosts = split.getBytes() / (BANDWIDTH_TO_SECONDS * processor.getCacheBandwidth().get(cacheLevel) * fromTeam.getCores());
            if (applyPenalty) {
                networkCosts += processor.getCacheLatency().get(cacheLevel) / LATENCY_TO_SECONDS;
            }
        } else if (distance == Teams.TeamDistance.DEVICE) {
            double streams = Double.min(team.getDevice().getMaxMainBandwidth() / team.getDevice().getMainBandwidth(), team.getCores());
            double teamStreams = streams * (team.getCores() / (double) team.getCores());
            networkCosts = split.getBytes() / (BANDWIDTH_TO_SECONDS * team.getDevice().getMainBandwidth() * teamStreams);
            if (applyPenalty) {
                networkCosts += fromTeam.getDevice().getMainLatency() / LATENCY_TO_SECONDS;
            }
        } else if (distance == Teams.TeamDistance.NODE) {
            Node fromNode = fromTeam.getDevice().getParent();

            String fromIdentifier = fromTeam.getDevice().getIdentifier();
            String teamIdentifier = team.getDevice().getIdentifier();
            networkCosts = split.getBytes() / (BANDWIDTH_TO_SECONDS * fromNode.getConnectivityBandwidth(fromIdentifier, teamIdentifier));
            if (applyPenalty) {
                networkCosts += fromNode.getConnectivityLatency(fromIdentifier, teamIdentifier) / LATENCY_TO_SECONDS;
            }
        } else {
            Node fromNode = fromTeam.getDevice().getParent();
            Node teamNode = team.getDevice().getParent();

            String fromIdentifier = fromNode.getIdentifier();
            String teamIdentifier = teamNode.getIdentifier();
            networkCosts = split.getBytes() / (BANDWIDTH_TO_SECONDS * network.getConnectivityBandwidth(fromIdentifier, teamIdentifier));
            if (applyPenalty) {
                networkCosts += network.getConnectivityLatency(fromIdentifier, teamIdentifier) / LATENCY_TO_SECONDS;
            }
        }

        return networkCosts;
    }

    @Override
    public double estimateAll(Set<DataSplit> splits, Team team, Set<Team> teams, Mapping history, Network network, boolean applyPenalty) {
        Table<DataSplit, Team, Team> dataSplitHolders = history.dataSplitHolder(splits, Sets.newHashSet(team), network);

        // Data without last accessor.
        long initialBytes = Sets.difference(splits, dataSplitHolders.rowKeySet()).stream().mapToLong(DataSplit::getBytes).sum();
        double streams = Double.min(team.getDevice().getMaxMainBandwidth() / team.getDevice().getMainBandwidth(), team.getCores());
        double teamStreams = streams * (team.getCores() / (double) team.getCores());
        double networkCosts = initialBytes / (BANDWIDTH_TO_SECONDS * team.getDevice().getMainBandwidth() * teamStreams);
        if (initialBytes > 0 && applyPenalty) {
            networkCosts += team.getDevice().getMainLatency() / LATENCY_TO_SECONDS;
        }

        // Data with last accessor.
        Map<Team, Set<DataSplit>> swapped = dataSplitHolders.column(team).entrySet().stream()
                .collect(Collectors.groupingBy(Map.Entry::getValue, Collectors.mapping(Map.Entry::getKey, Collectors.toSet())));
        for (Team fromTeam : swapped.keySet()) {
            long bytes = swapped.get(fromTeam).stream().mapToLong(DataSplit::getBytes).sum();
            Teams.TeamDistance distance = Teams.distance(fromTeam, team, network);

            if (distance == Teams.TeamDistance.PROCESSOR) {
                Processor processor = fromTeam.getProcessor();
                int cacheLevel = processor.getCacheBandwidth().size() - 1;
                networkCosts += bytes / (BANDWIDTH_TO_SECONDS * processor.getCacheBandwidth().get(cacheLevel) * fromTeam.getCores());
                if (applyPenalty) {
                    networkCosts += processor.getCacheLatency().get(cacheLevel) / LATENCY_TO_SECONDS;
                }
            } else if (distance == Teams.TeamDistance.DEVICE) {
                networkCosts += bytes / (BANDWIDTH_TO_SECONDS * team.getDevice().getMainBandwidth() * teamStreams);
                if (applyPenalty) {
                    networkCosts += team.getDevice().getMainLatency() / LATENCY_TO_SECONDS;
                }
            } else if (distance == Teams.TeamDistance.NODE) {
                Node fromNode = fromTeam.getDevice().getParent();
                String fromIdentifier = fromTeam.getDevice().getIdentifier();
                String teamIdentifier = team.getDevice().getIdentifier();

                long simultaneousStreams = teams.stream().filter(t -> t.getDevice().getIdentifier().equals(teamIdentifier)).count();

                networkCosts += bytes / (BANDWIDTH_TO_SECONDS * fromNode.getConnectivityBandwidth(fromIdentifier, teamIdentifier) / simultaneousStreams);
                if (applyPenalty) {
                    networkCosts += fromNode.getConnectivityLatency(fromIdentifier, teamIdentifier) / LATENCY_TO_SECONDS;
                }
            } else {
                Node fromNode = fromTeam.getDevice().getParent();
                Node teamNode = team.getDevice().getParent();

                String fromIdentifier = fromNode.getIdentifier();
                String teamIdentifier = teamNode.getIdentifier();
                networkCosts += bytes / (BANDWIDTH_TO_SECONDS * network.getConnectivityBandwidth(fromIdentifier, teamIdentifier));
                if (applyPenalty) {
                    networkCosts += network.getConnectivityLatency(fromIdentifier, teamIdentifier) / LATENCY_TO_SECONDS;
                }
            }
        }

        return networkCosts;
    }

    @Override
    public double latencyPenalty(Team from, Team to, Network network) {
        Teams.TeamDistance distance = Teams.distance(from, to, network);
        if (distance == Teams.TeamDistance.PROCESSOR) {
            Processor processor = from.getProcessor();
            int cacheLevel = processor.getCacheBandwidth().size() - 1;
            return processor.getCacheLatency().get(cacheLevel) / LATENCY_TO_SECONDS;
        } else if (distance == Teams.TeamDistance.DEVICE) {
            Device device = from.getDevice();
            return device.getMainLatency() / LATENCY_TO_SECONDS;
        } else if (distance == Teams.TeamDistance.NODE) {
            Node fromNode = from.getDevice().getParent();

            String fromIdentifier = from.getDevice().getIdentifier();
            String teamIdentifier = to.getDevice().getIdentifier();
            return fromNode.getConnectivityLatency(fromIdentifier, teamIdentifier) / LATENCY_TO_SECONDS;
        } else {
            Node fromNode = from.getDevice().getParent();
            Node teamNode = to.getDevice().getParent();

            String fromIdentifier = fromNode.getIdentifier();
            String teamIdentifier = teamNode.getIdentifier();
            return network.getConnectivityLatency(fromIdentifier, teamIdentifier) / LATENCY_TO_SECONDS;
        }
    }
}
