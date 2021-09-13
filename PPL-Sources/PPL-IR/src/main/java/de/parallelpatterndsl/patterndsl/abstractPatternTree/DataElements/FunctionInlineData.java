package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.SerialNode;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * Data element for encapsulating function calls, to allow for simplified function inlining during code generation.
 */
public class FunctionInlineData extends Data {

    /**
     * The function call defining this inlining variable.
     */
    private OperationExpression call;

    /**
     * The dimensionality of the variable.
     */
    private Integer dimension;

    /**
     * A String which is appended to the identifier in order to create a unique variable capturing the inlined result of the function call.
     */
    private String inlineEnding;

    /**
     * True, iff the function definition got inlined due to nested parallel patterns.
     */
    private boolean APTIsInlined = false;

    public FunctionInlineData(String identifier, PrimitiveDataTypes typeName, OperationExpression call, Integer dimension) {
        super(identifier, typeName, false);
        this.call = call;
        this.dimension = dimension;
    }

    public OperationExpression getCall() {
        return call;
    }

    public Integer getDimension() {
        return dimension;
    }

    @Override
    public long getBytes() {
        return 0;
    }

    @Override
    public Data createInlineCopy(String inlineIdentifier) {
        return new FunctionInlineData(getIdentifier() + "_" + inlineIdentifier, getTypeName(), getCall().simpleCopy(), getDimension());
    }

    public String getInlineEnding() {
        return inlineEnding;
    }

    public void setInlineEnding(String inlineEnding) {
        this.inlineEnding = inlineEnding;
    }

    public boolean isAPTIsInlined() {
        return APTIsInlined;
    }

    public void setAPTIsInlined(boolean APTIsInlined) {
        this.APTIsInlined = APTIsInlined;
    }

    /**
     * Gathers all array- and primitive data elements from a function inline data elements.
     * @return
     */
    public Set<Data> getNestedData() {
        HashSet<Data> result = new HashSet<>();

        for (Data nested: call.getOperands() ) {
            if (nested instanceof ArrayData || nested instanceof PrimitiveData) {
                result.add(nested);
            } else if (nested instanceof FunctionInlineData) {
                result.addAll(((FunctionInlineData) nested).getNestedData());
            }
        }
        return result;
    }

    /**
     * Returns the shape of the function corresponding with this call. Returns an empty array, iff the function returns a scalar value.
     * @return
     */
    public ArrayList<Integer> getShape() {
        FunctionNode node = AbstractPatternTree.getFunctionTable().get(call.getOperands().get(0).getIdentifier());
        SerialNode serialNode = (SerialNode) node;
        return serialNode.getShape();
    }

    /**
     * Creates a new expression based on itself by replacing the original data elements with new elements based on the inlineIdentifier.
     * This function is used to create copies of expression for APT inlining.
     *
     * @param globalVars
     * @param inlineIdentifier
     * @param variableTable
     * @return
     */
    public void createInlineCopies(ArrayList<Data> globalVars, String inlineIdentifier, HashMap<String, Data> variableTable) {
        call = call.createInlineCopy(globalVars, inlineIdentifier, variableTable);

        return;
    }

    /**
     * This function replaces a specific data element (oldData) with a new data element (newData) within the operands.
     * @param oldData
     * @param newData
     */
    public void replaceDataElement(Data oldData, Data newData) {
        call.replaceDataElement(oldData, newData);
    }
}
