package de.parallelpatterndsl.patterndsl.teams;

import com.google.common.collect.Sets;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Helper class for generating specific sets of teams.
 */
public class Teams {

    /**
     * Discrete distance measure of two teams.
     */
    public enum TeamDistance {
        PROCESSOR,
        DEVICE,
        NODE,
        NETWORK;
    }

    /**
     * Estimates the distance between two teams with respect to a network.
     * @param t1 - team 1.
     * @param t2 - team 2.
     * @param target - network.
     * @return TeamDistance enum.
     */
    public static TeamDistance distance(Team t1, Team t2, Network target) {
        if (t1.getDevice().equals(t2.getDevice())) {
            if (t1.getProcessor().equals(t2.getProcessor())) {
                return TeamDistance.PROCESSOR;
            } else {
                return TeamDistance.DEVICE;
            }
        } else {
            Node n1 = t1.getDevice().getParent();
            Node n2 = t2.getDevice().getParent();

            if (n1.equals(n2)) {
                return TeamDistance.NODE;
            } else {
                return TeamDistance.NETWORK;
            }
        }
    }

    /**
     * A local move, where the core count of each team is scaled up according to the scaling factor.
     * @param teams - set of teams.
     * @param factor - scaling factor.
     * @return set of scaled teams.
     */
    public static Set<Team> scaleUp(Set<Team> teams, int factor) {
        Map<Processor, Integer> occupationMap = new HashMap<>();
        for (Team team : teams) {
            Processor processor = team.getProcessor();
            int occupied = occupationMap.getOrDefault(processor, 0);
            occupied += team.getCores();

            occupationMap.put(processor, occupied);
        }


        Set<Team> newTeams = new HashSet<>();
        for (Team team : teams) {
            Processor processor = team.getProcessor();
            int occupation = occupationMap.get(processor);
            int free = processor.getCores() - occupation;
            int addCores = Integer.min(free, team.getCores() * (factor - 1));

            Team newTeam = new Team(team);
            newTeam.setCores(team.getCores() + addCores);
            occupationMap.put(processor, occupation + addCores);
            newTeams.add(newTeam);
        }

        return newTeams;
    }

    /**
     * A local move, where the core count of each team is scaled down according to the scaling factor.
     * @param teams - set of teams.
     * @param factor - scaling factor.
     * @return set of scaled teams.
     */
    public static Set<Team> scaleDown(Set<Team> teams, int factor) {
        HashSet<Team> newTeams = new HashSet<>();
        for (Team team : teams) {
            Team newTeam = new Team(team);
            newTeam.setCores(Integer.max(1, team.getCores() / factor));

            newTeams.add(newTeam);
        }

        return newTeams;
    }

    /**
     * A move, which is either local or a jump.
     * In case of a local move, the number of teams is doubled while the number of devices is kept constant.
     * Hence, the new teams on sockets and streaming multiprocessors are added, where the CPUs and GPUs are aöready in use.
     * In case of a jump, a new node is searched for and teams for every socket of the node are added.
     * @param teams - current set of teams.
     * @param local - controls local move or jump.
     * @param target - network.
     * @return spreaded set of teams.
     */
    public static Set<Team> spread(Set<Team> teams, boolean local, Network target) {
        Set<Team> newTeams = new HashSet<>();
        if (local) {
            HashMap<Device, HashSet<Processor>> inUse = new HashMap<>();
            for (Team team : teams) {
                if (!inUse.containsKey(team.getDevice())) {
                    inUse.put(team.getDevice(), new HashSet<>());
                }

                inUse.get(team.getDevice()).add(team.getProcessor());
            }

            for (Team team : teams) {
                for (Processor p : team.getDevice().getProcessor()) {
                    if (!inUse.get(team.getDevice()).contains(p)) {
                        Team newTeam = new Team(team.getDevice(), p, team.getCores());
                        newTeams.add(newTeam);
                        inUse.get(team.getDevice()).add(p);
                        break;
                    }
                }
            }
        } else {
            Set<Device> devices = teams.stream().map(Team::getDevice).collect(Collectors.toSet());
            Optional<Node> freeNode = target.getNodes().stream().filter(n -> Sets.intersection(devices, new HashSet<>(n.getDevices())).isEmpty()).findAny();
            if (freeNode.isPresent()) {
                Node free = freeNode.get();
                for (Device device : free.getDevices()) {
                    if (device.getType().equals("GPU")) {
                        continue;
                    }

                    for (Processor processor : device.getProcessor()) {
                        Team newTeam = new Team(device, processor, processor.getCores());
                        newTeams.add(newTeam);
                    }
                }
            }
        }

        if (newTeams.size() > 0) {
            newTeams.addAll(teams.stream().map(Team::new).collect(Collectors.toSet()));
        }
        return newTeams.size() > 0 ? newTeams : null;
    }

    /**
     * A jump, where count many teams on additional streaming multiprocessors are created.
     * Thereby, local streaming multiprocessors are preferred over those on new GPUs.
     * @param teams - current set of teams.
     * @param count - number of added streaming multiprocessor.
     * @param target - network.
     * @return new set of teams.
     */
    public static Set<Team> offloading(Set<Team> teams, int count, Network target) {
        Set<Team> newTeams = new HashSet<>();

        // Existing devices.
        Set<Processor> usedProcessor = teams.stream().filter(t -> t.getDevice().getType().equals("GPU")).map(Team::getProcessor).collect(Collectors.toSet());
        Set<Device> usedDevices = teams.stream().map(Team::getDevice).collect(Collectors.toSet());
        int i = count;
        for (Device device : usedDevices) {
            if (!device.getType().equals("GPU")) {
                continue;
            }

            LinkedList<Processor> freeProcessor = new LinkedList<>(device.getProcessor());
            freeProcessor.removeAll(usedProcessor);
            Iterator<Processor> iter = freeProcessor.iterator();
            for (; i > 0 && iter.hasNext(); i--) {
                Processor p = iter.next();
                Team newTeam = new Team(device, p, p.getCores());
                newTeams.add(newTeam);
            }

            if (i == 0) {
                newTeams.addAll(teams.stream().map(Team::new).collect(Collectors.toSet()));
                return newTeams;
            }
        }
        if (i < count) {
            newTeams.addAll(teams.stream().map(Team::new).collect(Collectors.toSet()));
            return newTeams;
        }

        // Else: new device.
        for (Node node : target.getNodes()) {
            Set<Device> freeDevices = new HashSet<>(node.getDevices());
            freeDevices.removeAll(usedDevices);

            if (freeDevices.size() < node.getDevices().size() && !freeDevices.isEmpty()) {
                for (Device device : freeDevices) {
                    if (!device.getType().equals("GPU")) {
                        continue;
                    }

                    Iterator<Processor> iter = device.getProcessor().iterator();
                    for (; i > 0 && iter.hasNext(); i--) {
                        Processor p = iter.next();
                        Team newTeam = new Team(device, p, p.getCores());
                        newTeams.add(newTeam);
                    }

                    if (i == 0) {
                        newTeams.addAll(teams.stream().map(Team::new).collect(Collectors.toSet()));
                        return newTeams;
                    }
                }
            }
        }

        if (newTeams.size() > 0) {
            newTeams.addAll(teams.stream().map(Team::new).collect(Collectors.toSet()));
        }
        return newTeams.size() > 0 ? newTeams : null;
    }

    /**
     * Creates an initial set of teams consisting of a single CPU team on the first node.
     * @param target - network.
     * @return singleton set of teams.
     */
    public static Set<Team> initialTeams(Network target) {
        // Initial hypotheses: Currently a single.
        Set<Team> teams = new HashSet<>();
        Device device = target.getNodes().get(0).getDevices().stream().filter(d -> d.getType().equals("CPU")).findFirst().get();
        Processor processor = device.getProcessor().get(0);
        Team team = new Team(device, processor, 1);
        teams.add(team);

        return teams;
    }

    /**
     * Creates a host team for a GPU.
     * @param device - GPU.
     * @param network - network.
     * @return CPU team.
     */
    public static Team host(Device device, Network network) {
        Node node = device.getParent();
        Device host = null;
        for (Device d : node.getDevices()) {
            if (d.getType().equals("CPU")) {
                host = d;
            }
        }

        Team hostTeam = new Team(host, host.getProcessor().get(0), host.getProcessor().get(0).getCores());
        return hostTeam;
    }

}
