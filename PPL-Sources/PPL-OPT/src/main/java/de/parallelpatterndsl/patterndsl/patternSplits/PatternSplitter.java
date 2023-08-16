package de.parallelpatterndsl.patterndsl.patternSplits;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaValue;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplitTable;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.MapDataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.StencilDataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaList;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.performance.simple.SimpleParallelismEstimator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The PatternSplitter provides static methods for splitting a ParallelCallNode object based on the parallel pattern.
 */
public class PatternSplitter {

    private static final SimpleParallelismEstimator parallelismEstimator = new SimpleParallelismEstimator();

    /**
     * Splits the ParallelCallNode based on the parallel pattern.
     * In case of recurrences, several levels of splits are created.
     * @param callNode - ParallelCallNode to be split.
     * @param patternSplitSize - hyperparameter pattern split size.
     * @param dataSplitSize - hyperparameter data split size.
     * @return list of sets of splits.
     */
    public static List<HashSet<ParallelPatternSplit>> split(ParallelCallNode callNode, int patternSplitSize, int dataSplitSize) {
        FunctionNode pattern = AbstractPatternTree.getFunctionTable().get(callNode.getFunctionIdentifier());
        if (pattern instanceof MapNode) {
            return PatternSplitter.splitMap(callNode, patternSplitSize, dataSplitSize);
        } else if (pattern instanceof StencilNode) {
            return PatternSplitter.splitStencil(callNode, patternSplitSize, dataSplitSize);
        } else if (pattern instanceof ReduceNode) {
            return PatternSplitter.splitReduction(callNode, patternSplitSize, dataSplitSize);
        } else if (pattern instanceof DynamicProgrammingNode) {
            return PatternSplitter.splitDynamicProgramming(callNode, patternSplitSize, dataSplitSize);
        }

        return null;
    }

    /**
     * Splits a ParallelCallNode object, which corresponds to a map pattern.
     * For more details see Thesis "Global Optimization of Parallel Pattern-based Algorithms for Heterogeneous Architectures" in git.
     * @param node - ParallelCallNode to be split.
     * @param patternSplitSize - hyperparameter pattern split size.
     * @param dataSplitSize - hyperparameter data split size.
     * @return list of set of splits (single level).
     */
    private static List<HashSet<ParallelPatternSplit>> splitMap(ParallelCallNode node, int patternSplitSize, int dataSplitSize) {
        ParallelNode pattern = (ParallelNode) AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
        long[] lengths = parallelismEstimator.estimate(node);
        ArrayList<Long> starts = new ArrayList<>();
        starts.add(((MetaValue<Long>) node.getAdditionalArguments().get(1)).getValue());

        ArrayList<HashSet<ParallelPatternSplit>> nodeJobs = new ArrayList<>(lengths.length);
        for (int t = 0; t < lengths.length; t++) {
            HashSet<ParallelPatternSplit> jobsAtDepth = new HashSet<>();

            // 1. Create jobs.
            for (int j = 0; j < lengths[0]; j = j + patternSplitSize) {
                long[] startIndices = new long[1];
                startIndices[0] = j + starts.get(0);

                long[] patternSplitSizes = new long[1];
                patternSplitSizes[0] = Long.min(patternSplitSize, lengths[0] - j);

                ParallelPatternSplit job = new ParallelPatternSplit(node, startIndices, patternSplitSizes);
                jobsAtDepth.add(job);
            }

            // 2. Add Input network packages.
            for (int k = 0; k < node.getInputElements().size(); k++) {
                Data inputData = node.getInputElements().get(k);
                Data argumentData = pattern.getArgumentValues().get(k);
                if (inputData instanceof PrimitiveData) {
                    for (ParallelPatternSplit job : jobsAtDepth) {
                        job.addInputNetworkPackage(DataSplitTable.get((PrimitiveData) inputData));
                    }
                } else if (inputData instanceof ArrayData) {
                    for (DataAccess access : argumentData.getTrace().getDataAccesses()) {
                        for (ParallelPatternSplit job : jobsAtDepth) {
                            if (!(access instanceof MapDataAccess)) {
                                job.addAllInputNetworkPackages(DataSplitTable.get((ArrayData) inputData, 0, ((ArrayData) inputData).getShape().get(0)));
                                continue;
                            }

                            MapDataAccess mapAccess = (MapDataAccess) access;
                            long startIndex = mapAccess.getScalingFactor() * job.getStartIndices()[0] + mapAccess.getShiftOffset();
                            long endIndex = (long) (mapAccess.getScalingFactor() * (job.getStartIndices()[0] + job.getLengths()[0] - 1) + mapAccess.getShiftOffset());
                            long padding = dataSplitSize - endIndex % dataSplitSize;
                            endIndex += padding - 1;
                            int stepSize = Integer.max(mapAccess.getScalingFactor(), dataSplitSize);

                            for (long index = startIndex; index <= endIndex; index += stepSize) {
                                job.addInputNetworkPackage(DataSplitTable.get((ArrayData) inputData, index));
                            }
                        }
                    }
                }
            }

            // 3. Add output network packages.
            Data outputData = node.getOutputElements().get(0);
            Data returnData = pattern.getReturnElement();
            if (outputData instanceof PrimitiveData) {
                for (ParallelPatternSplit job : jobsAtDepth) {
                    job.addInputNetworkPackage(DataSplitTable.get((PrimitiveData) outputData));
                }
            } else if (outputData instanceof ArrayData) {
                for (DataAccess access : returnData.getTrace().getDataAccesses()) {
                    if (!(access instanceof MapDataAccess)) {
                        continue;
                    }

                    MapDataAccess mapAccess = (MapDataAccess) access;
                    for (ParallelPatternSplit job : jobsAtDepth) {
                        long startIndex = mapAccess.getScalingFactor() * job.getStartIndices()[0] + mapAccess.getShiftOffset();
                        long endIndex = (int) (mapAccess.getScalingFactor() * (job.getStartIndices()[0] + job.getLengths()[0] - 1) + mapAccess.getShiftOffset());
                        long padding = dataSplitSize - endIndex % dataSplitSize;
                        endIndex += padding - 1;
                        long stepSize = Integer.max(mapAccess.getScalingFactor(), dataSplitSize);

                        for (long index = startIndex; index <= endIndex; index += stepSize) {
                            job.addOutputNetworkPackage(DataSplitTable.get((ArrayData) outputData, index));
                        }
                    }
                }
            }

            nodeJobs.add(jobsAtDepth);
        }

        return nodeJobs;
    }


    /**
     * Splits a ParallelCallNode object, which corresponds to a stencil pattern.
     * For more details see Thesis "Global Optimization of Parallel Pattern-based Algorithms for Heterogeneous Architectures" in git.
     * @param node - ParallelCallNode to be split.
     * @param patternSplitSize - hyperparameter pattern split size.
     * @param dataSplitSize - hyperparameter data split size.
     * @return list of set of splits (single level).
     */
    private static List<HashSet<ParallelPatternSplit>> splitStencil(ParallelCallNode node, int patternSplitSize, int dataSplitSize) {
        ParallelNode pattern = (ParallelNode) AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
        long[] lengths = parallelismEstimator.estimate(node);
        ArrayList<Long> starts = new ArrayList<>(((MetaList<Long>) node.getAdditionalArguments().get(1)).getValues());

        ArrayList<HashSet<ParallelPatternSplit>> nodeJobs = new ArrayList<>(1);
        HashSet<ParallelPatternSplit> jobs = new HashSet<>();

        // 1. Create jobs.
        for (int j = 0; j < lengths[0]; j = j + patternSplitSize) {
            long[] startIndices = new long[lengths.length];
            startIndices[0] = j + starts.get(0);
            long[] jobLenghts = Arrays.copyOf(lengths, lengths.length);
            jobLenghts[0] = Long.min(patternSplitSize, lengths[0] - j);

            for (int i = 1; i < lengths.length; i++)
            {
                startIndices[i] = starts.get(i);
                jobLenghts[i] = lengths[i];
            }

            ParallelPatternSplit job = new ParallelPatternSplit(node, startIndices, jobLenghts);
            jobs.add(job);
        }

        // 2. Add Input network packages.
        for (int k = 0; k < node.getInputElements().size(); k++) {
            Data inputData = node.getInputElements().get(k);
            Data argumentData = pattern.getArgumentValues().get(k);
            if (inputData instanceof PrimitiveData) {
                for (ParallelPatternSplit job : jobs) {
                    job.addInputNetworkPackage(DataSplitTable.get((PrimitiveData) inputData));
                }
            } else if (inputData instanceof ArrayData) {
                for (DataAccess access : argumentData.getTrace().getDataAccesses()) {
                    for (ParallelPatternSplit job : jobs) {
                        if (!(access instanceof StencilDataAccess)) {
                            job.addAllInputNetworkPackages(DataSplitTable.get((ArrayData) inputData, 0, ((ArrayData) inputData).getShape().get(0)));
                            continue;
                        }

                        StencilDataAccess stencilAccess = (StencilDataAccess) access;
                        long startIndex = stencilAccess.getScalingFactors().get(0) * job.getStartIndices()[0] + stencilAccess.getShiftOffsets().get(0);
                        long endIndex = (long) (stencilAccess.getScalingFactors().get(0) * (job.getStartIndices()[0] + job.getLengths()[0] - 1) + stencilAccess.getShiftOffsets().get(0));
                        long padding = dataSplitSize - endIndex % dataSplitSize;
                        endIndex += padding - 1;
                        long stepSize = Integer.max(stencilAccess.getScalingFactors().get(0), dataSplitSize);

                        for (long index = startIndex; index <= endIndex; index += stepSize) {
                            job.addInputNetworkPackage(DataSplitTable.get((ArrayData) inputData, index));
                        }
                    }
                }
            }
        }

        // 3. Add output network packages.
        Data outputData = node.getOutputElements().get(0);
        Data returnData = pattern.getReturnElement();
        if (outputData instanceof PrimitiveData) {
            for (ParallelPatternSplit job : jobs) {
                job.addInputNetworkPackage(DataSplitTable.get((PrimitiveData) outputData));
            }
        } else if (outputData instanceof ArrayData) {
            for (DataAccess access : returnData.getTrace().getDataAccesses()) {
                if (!(access instanceof StencilDataAccess)) {
                    continue;
                }

                StencilDataAccess stencilAccess = (StencilDataAccess) access;
                for (ParallelPatternSplit job : jobs) {
                    long startIndex = stencilAccess.getScalingFactors().get(0) * job.getStartIndices()[0] + stencilAccess.getShiftOffsets().get(0);
                    long endIndex = (long) (stencilAccess.getScalingFactors().get(0) * (job.getStartIndices()[0] + job.getLengths()[0] - 1) + stencilAccess.getShiftOffsets().get(0));
                    long padding = dataSplitSize - endIndex % dataSplitSize;
                    endIndex += padding - 1;
                    long stepSize = Integer.max(stencilAccess.getScalingFactors().get(0), dataSplitSize);

                    for (long index = startIndex; index <= endIndex; index += stepSize) {
                        job.addOutputNetworkPackage(DataSplitTable.get((ArrayData) outputData, index));
                    }
                }
            }
        }

        nodeJobs.add(jobs);
        return nodeJobs;
    }

    /**
     * Splits a ParallelCallNode object, which corresponds to a reduction pattern.
     * For more details see Thesis "Global Optimization of Parallel Pattern-based Algorithms for Heterogeneous Architectures" in git.
     * @param node - ParallelCallNode to be split.
     * @param patternSplitSize - hyperparameter pattern split size.
     * @param dataSplitSize - hyperparameter data split size.
     * @return list of set of splits (levels according to depth of recurrences).
     */
    private static List<HashSet<ParallelPatternSplit>> splitReduction(ParallelCallNode node, int patternSplitSize, int dataSplitSize) {
        ParallelNode pattern = (ParallelNode) AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
        long[] lengths = parallelismEstimator.estimate(node);
        long arity = ((MetaList<Long>) node.getAdditionalArguments().get(0)).getValues().get(1);
        long bytes = node.getOutputElements().get(0).getBytes();
        ArrayList<Long> starts = new ArrayList<>();
        starts.add(((MetaList<Long>) node.getAdditionalArguments().get(0)).getValues().get(3));

        ArrayList<ArrayList<ParallelPatternSplit>> nodeJobs = new ArrayList<>(lengths.length);
        long operations = lengths[0];
        // 1. Create jobs.
        ArrayList<ParallelPatternSplit> jobsAtDepth = new ArrayList<>();
        for (int j = 0; j < operations; j = j + patternSplitSize) {
            long[] startIndices = new long[1];
            startIndices[0] = j + starts.get(0);

            long[] patternSplitSizes = new long[1];
            patternSplitSizes[0] = Long.min(patternSplitSize, operations - j);

            ParallelPatternSplit job = new ParallelPatternSplit(node, startIndices, patternSplitSizes);
            jobsAtDepth.add(job);
        }

        // 2. Add Input network packages.
        for (int k = 0; k < node.getInputElements().size(); k++) {
            Data inputData = node.getInputElements().get(k);
            Data argumentData = pattern.getArgumentValues().get(k);
            if (inputData instanceof PrimitiveData) {
                for (ParallelPatternSplit job : jobsAtDepth) {
                    job.addInputNetworkPackage(DataSplitTable.get((PrimitiveData) inputData));
                }
            } else if (inputData instanceof ArrayData) {
                for (DataAccess access : argumentData.getTrace().getDataAccesses()) {
                    for (ParallelPatternSplit job : jobsAtDepth) {
                        if (!(access instanceof MapDataAccess)) {
                            job.addAllInputNetworkPackages(DataSplitTable.get((ArrayData) inputData, 0, ((ArrayData) inputData).getShape().get(0)));
                            continue;
                        }

                        MapDataAccess mapAccess = (MapDataAccess) access;
                        long startIndex = mapAccess.getScalingFactor() * job.getStartIndices()[0] + mapAccess.getShiftOffset();
                        long endIndex = (long) (mapAccess.getScalingFactor() * (job.getStartIndices()[0] + job.getLengths()[0] - 1) + mapAccess.getShiftOffset());
                        long padding = dataSplitSize - endIndex % dataSplitSize;
                        endIndex += padding - 1;
                        long stepSize = Integer.max(mapAccess.getScalingFactor(), dataSplitSize);

                        for (long index = startIndex; index <= endIndex; index += stepSize) {
                            job.addInputNetworkPackage(DataSplitTable.get((ArrayData) inputData, index));
                        }
                    }
                }
            }
        }

        // 3. Add output network packages.
        Data outputData = node.getOutputElements().get(0);
        ParallelPatternSplit lastJob = jobsAtDepth.iterator().next();
        lastJob.addOutputNetworkPackage(DataSplitTable.get((PrimitiveData) outputData));


        nodeJobs.add(jobsAtDepth);


        return nodeJobs.stream().map(HashSet::new).collect(Collectors.toList());
    }


    /**
     * Splits a ParallelCallNode object, which corresponds to a dynamic programming pattern.
     * For more details see Thesis "Global Optimization of Parallel Pattern-based Algorithms for Heterogeneous Architectures" in git.
     * @param node - ParallelCallNode to be split.
     * @param patternSplitSize - hyperparameter pattern split size.
     * @param dataSplitSize - hyperparameter data split size.
     * @return list of set of splits (levels according to depth of dynamic programming).
     */
    private static List<HashSet<ParallelPatternSplit>> splitDynamicProgramming(ParallelCallNode node, int patternSplitSize, int dataSplitSize) {
        ParallelNode pattern = (ParallelNode) AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
        long[] lengths = parallelismEstimator.estimate(node);
        ArrayList<Long> starts = new ArrayList<>(((MetaList<Long>) node.getAdditionalArguments().get(1)).getValues());

        ArrayList<ArrayList<ParallelPatternSplit>> nodeJobs = new ArrayList<>(lengths.length);
        for (int t = 0; t < lengths.length; t++) {
            ArrayList<ParallelPatternSplit> jobsAtDepth = new ArrayList<>();
            for (long j = 0; j < lengths[t]; j = j + patternSplitSize) {
                long[] startIndices = new long[2];
                startIndices[0] = t;
                startIndices[1] = j + starts.get(1);

                long[] patternSplitSizes = new long[1];
                patternSplitSizes[0] = Long.min(patternSplitSize, lengths[0] - j);

                ParallelPatternSplit job = new ParallelPatternSplit(node, startIndices, patternSplitSizes);
                jobsAtDepth.add(job);
            }

            nodeJobs.add(jobsAtDepth);
        }

        return nodeJobs.stream().map(HashSet::new).collect(Collectors.toList());
    }


}
