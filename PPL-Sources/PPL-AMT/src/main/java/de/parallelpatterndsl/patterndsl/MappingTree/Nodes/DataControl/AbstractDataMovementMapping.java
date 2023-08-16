package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.MainMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Optional;

public abstract class AbstractDataMovementMapping extends MappingNode {
    public AbstractDataMovementMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, PatternNode node) {
        super(parent, variableTable, node);
    }

    public AbstractDataMovementMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, ArrayList<Data> inputElements, ArrayList<Data> outputElements, ArrayList<DataAccess> inputAccesses, ArrayList<DataAccess> outputAccesses) {
        super(parent, variableTable, inputElements, outputElements, inputAccesses, outputAccesses);
    }
}
