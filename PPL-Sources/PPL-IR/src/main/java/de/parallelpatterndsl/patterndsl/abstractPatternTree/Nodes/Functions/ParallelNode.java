package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveDataTypes;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;

/**
 * Abstract definition of parallel pattern nodes.
 */
public abstract class ParallelNode extends FunctionNode {

    /**
     * The Data element, which will be returned at the end of parallel node.
     */
    private Data returnElement;

    public ParallelNode(String identifier) {
        super(identifier);
    }

    public Data getReturnElement() {
        return returnElement;
    }

    public void setReturnElement(Data returnElement) {
        this.returnElement = returnElement;
    }

    @Override
    public boolean returnsArray() {
        if (returnElement instanceof ArrayData) {
            return true;
        }
        return false;
    }

    @Override
    public PrimitiveDataTypes getReturnType() {
        return returnElement.getTypeName();
    }
}
