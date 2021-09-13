package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.EndPoint;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.CallNode;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;

import java.util.*;

public class SerializedParallelCallMapping extends ParallelCallMapping {

    public SerializedParallelCallMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, CallNode aptNode, ArrayList<Long> startIndex, ArrayList<Long> numIterations, Processor executor, int numThreads) {
        super(parent, variableTable, aptNode, startIndex, numIterations, executor, numThreads, Optional.empty(), new HashSet<>());
    }

    @Override
    public HashSet<DataPlacement> getNecessaryData() {
        HashSet<DataPlacement> result = new HashSet<>();

        for (Data inputs: getInputElements() ) {
            ArrayList<EndPoint> partial = new ArrayList<>();
            EndPoint element;
            if (inputs instanceof ArrayData) {
                element = new EndPoint(AbstractMappingTree.getDefaultDevice(), 0, ((ArrayData) inputs).getShape().get(0),  new HashSet<>(), false);
            } else {
                element = new EndPoint(AbstractMappingTree.getDefaultDevice(), 0, 1, new HashSet<>(), false);
            }
            partial.add(element);
            DataPlacement placement = new DataPlacement(partial, inputs);
            result.add(placement);
        }
        return result;
    }

    @Override
    public HashSet<DataPlacement> getOutputData() {
        HashSet<DataPlacement> result = new HashSet<>();

        for (Data outputs: getOutputElements() ) {
            ArrayList<EndPoint> partial = new ArrayList<>();
            EndPoint element;
            if (outputs instanceof ArrayData) {
                element = new EndPoint(AbstractMappingTree.getDefaultDevice(), 0, ((ArrayData) outputs).getShape().get(0), new HashSet<>(), false);
            } else {
                element = new EndPoint(AbstractMappingTree.getDefaultDevice(), 0, 1, new HashSet<>(), false);
            }
            partial.add(element);
            DataPlacement placement = new DataPlacement(partial, outputs);
            result.add(placement);
        }
        return result;
    }
}
