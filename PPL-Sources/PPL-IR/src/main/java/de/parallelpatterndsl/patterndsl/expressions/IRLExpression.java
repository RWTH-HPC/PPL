package de.parallelpatterndsl.patterndsl.expressions;

import de.parallelpatterndsl.patterndsl.PatternTypes;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Abstract class that allows to store both Assignment and Operation expressions without differentiating.
 */
public abstract class IRLExpression {


    /**
     * Extracts the data accesses from the expressions.
     * @param patternType
     * @return
     */
    public abstract ArrayList<DataAccess> getDataAccesses(PatternTypes patternType);

    /**
     * Returns the number of operations within the expression.
     * @return
     */
    public abstract int getOperationCount();

    /**
     * Gets the shape of an expression. Returns an empty list, iff the expression results in a scalar value.
     * @return
     */
    public abstract ArrayList<Integer> getShape();

    /**
     * Returns true if the expression contains an IO access
     * @return
     */
    public abstract boolean isHasIOData();

    /**
     * Creates a new expression based on itself by replacing the original data elements with new elements based on the inlineIdentifier.
     * This function is used to create copies of expression for APT inlining.
     *
     * @param globalVars
     * @param inlineIdentifier
     * @param variableTable
     * @return
     */
    public abstract IRLExpression createInlineCopy(ArrayList<Data> globalVars, String inlineIdentifier, HashMap<String,Data> variableTable);


    /**
     * This function replaces a specific data element (oldData) with a new data element (newData) within the operands.
     * @param oldData
     * @param newData
     */
    public abstract void replaceDataElement(Data oldData, Data newData);
}
