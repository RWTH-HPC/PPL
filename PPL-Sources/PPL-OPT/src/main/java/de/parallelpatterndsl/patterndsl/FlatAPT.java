package de.parallelpatterndsl.patterndsl;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ComplexExpressionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.SimpleExpressionBlockNode;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplit;
import de.parallelpatterndsl.patterndsl.dataSplits.SyncDataSplit;
import de.parallelpatterndsl.patterndsl.dataSplits.TempDataSplit;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.patternSplits.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * FlatAPT represents a synchronization efficient APT in form of table.
 */
public class FlatAPT{

    private final ArrayList<HashSet<PatternSplit>> table;

    private final int patternSplitSize;

    private final int dataSplitSize;

    FlatAPT(int patternSplitSize, int dataSplitSize) {
        this.table = new ArrayList<>();
        this.patternSplitSize = patternSplitSize;
        this.dataSplitSize = dataSplitSize;
    }

    /**
     * Returns the number of steps of the FlatAPT.
     * @return size
     */
    public int size() {
        return this.table.size();
    }

    /**
     * Returns the pattern splits of a step.
     * @param step
     * @return Set
     */
    public Set<PatternSplit> getSplits(int step) {
        return Collections.unmodifiableSet(this.table.get(step));
    }

    //================================================================================
    // Construct
    //================================================================================

    private enum DataDependency {
        SYNCHRONIZATION,
        READ_AFTER_WRITE,
        WRITE_AFTER_READ,
        WRITE_AFTER_WRITE;
    }

    /**
     * Adds a ParallelCallNode to the FlatAPT by analyzing the dependencies and bubbling up until correct step.
     * @param node - ParallelCallNode
     */
    void add(ParallelCallNode node) {
        List<HashSet<ParallelPatternSplit>> patternSplits = PatternSplitter.split(node, patternSplitSize, dataSplitSize);

        for (Set<ParallelPatternSplit> splits : patternSplits) {
            int step = this.table.size();
            while(step > 0) {
                Set<DataSplit> inputData = splits.stream().flatMap(j -> j.getInputDataSplits().stream()).collect(Collectors.toSet());
                Set<DataSplit> outputData = splits.stream().flatMap(j -> j.getOutputDataSplits().stream()).collect(Collectors.toSet());
                HashMap<DataSplit, DataDependency> relation = happensBefore(step - 1, inputData, outputData);
                resolveDependencies(relation, node);

                if (!relation.isEmpty()) {
                    break;
                } else {
                    --step;
                }
            }

            if (step == this.table.size()) {
                HashSet<PatternSplit> newStep = new HashSet<>();
                newStep.addAll(splits);
                this.table.add(newStep);
            } else {
                this.table.get(step).addAll(splits);
            }
        }
    }

    /**
     * Adds a PatternNode to the FlatAPT by analyzing the dependencies and bubbling up until correct step.
     * @param node - PatternNode
     */
    void add(PatternNode node) {
        if (node instanceof ParallelCallNode) {
            this.add((ParallelCallNode) node);
            return;
        }

        SerialPatternSplit job;
        if (node instanceof SimpleExpressionBlockNode) {
            SimpleExpressionBlockNode simpleNode = (SimpleExpressionBlockNode) node;
            if (simpleNode.getExpressionList().stream().anyMatch(IRLExpression::isHasIOData)) {
                job = new IOPatternSplit(node);
            } else {
                job = new SerialPatternSplit(node);
            }
        } else if (node instanceof ComplexExpressionNode) {
            ComplexExpressionNode complexNode = (ComplexExpressionNode) node;
            if (complexNode.getExpression().isHasIOData()) {
                job = new IOPatternSplit(node);
            } else {
                job = new SerialPatternSplit(node);
            }
        } else {
            job = new SerialPatternSplit(node);
        }

        int step = this.table.size();
        while(step > 0 ) {
            HashMap<DataSplit, DataDependency> relation = happensBefore(step - 1, job.getInputDataSplits(), job.getOutputDataSplits());
            resolveDependencies(relation, node);

            if (!relation.isEmpty() || node.containsSynchronization()) {
                break;
            } else {
                --step;
            }
        }

        if (step == this.table.size()) {
            HashSet<PatternSplit> newStep = new HashSet<>();
            newStep.add(job);
            this.table.add(newStep);
        } else {
            this.table.get(step).add(job);
        }
    }

    /**
     * Implements the happensBefore relation in form of RAW, WAW, WAR dependencies, see Hennessy et. al. (Thesis).
     * @param step - step to be compared to.
     * @param inputData - input data splits.
     * @param outputData - output data splits.
     * @return relation
     */
    private HashMap<DataSplit, DataDependency> happensBefore(int step, Set<DataSplit> inputData, Set<DataSplit> outputData) {
        HashMap<DataSplit, DataDependency> relation = new HashMap<>();
        HashSet<PatternSplit> patternSplits = this.table.get(step);

        Set<DataSplit> stepInput = patternSplits.stream().flatMap(j -> j.getInputDataSplits().stream()).collect(Collectors.toSet());
        Set<DataSplit> stepOutput = patternSplits.stream().flatMap(j -> j.getOutputDataSplits().stream()).collect(Collectors.toSet());

        Set<DataSplit> readAfterWrite = stepOutput.stream()
                .filter(inputData::contains)
                .collect(Collectors.toSet());

        Set<DataSplit> writeAfterRead = stepInput.stream()
                .filter(outputData::contains)
                .collect(Collectors.toSet());

        Set<DataSplit> writeAfterWrite = stepOutput.stream()
                .filter(outputData::contains)
                .collect(Collectors.toSet());

        for (DataSplit data : writeAfterRead) {
            relation.put(data, DataDependency.WRITE_AFTER_READ);
        }

        for (DataSplit data : writeAfterWrite) {
            relation.put(data, DataDependency.WRITE_AFTER_WRITE);
        }

        for (DataSplit data : readAfterWrite) {
            relation.put(data, DataDependency.READ_AFTER_WRITE);
        }
        if (patternSplits.stream().filter(x -> x instanceof SerialPatternSplit).anyMatch(x -> ((SerialPatternSplit) x).getNode().containsSynchronization())) {
            relation.put(new SyncDataSplit(), DataDependency.SYNCHRONIZATION);
        }

        return relation;
    }

    /**
     * Resolves WAW, WAR dependencies by renaming.
     * @param relation - happensBefore relation.
     * @param node - PatternNode of relation.
     */
    private void resolveDependencies(HashMap<DataSplit, DataDependency> relation, PatternNode node) {
        Iterator<Map.Entry<DataSplit, DataDependency>> iter = relation.entrySet().iterator();
        while (iter.hasNext()) {
            Map.Entry<DataSplit, DataDependency> dependency = iter.next();

            if (dependency.getKey() instanceof TempDataSplit) {
                continue;
            }
            if (dependency.getKey() instanceof SyncDataSplit) {
                break;
            }
            switch (dependency.getValue()) {
                case READ_AFTER_WRITE:
                    break;
                case WRITE_AFTER_READ:
                case WRITE_AFTER_WRITE:
                    int index = dependency.getKey().getData().getTrace().getAccessingNodes().indexOf(node);
                    if (!dependency.getKey().getData().getCopyIndices().contains(index)) {
                        dependency.getKey().getData().createCopy(index);
                    }

                    //iter.remove();
                    break;
            }
        }
    }
    
}
