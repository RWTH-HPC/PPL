package de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements;

import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;


/**
 * An extension of the data element to handle the additional information for an array.
 */
public class ArrayData extends Data{

    /**
     * The shape of the data element. e.g. [2,3] a matrix with 2 rows and 3 columns.
     */
    private ArrayList<Integer> shape;

    /**
     * True, iff the array is stored on the heap.
     */
    private boolean onStack;

    /**
     * True, iff the array is part of another array.
     */
    private boolean isLocalPointer;

    public ArrayData(String identifier, PrimitiveDataTypes typeName, boolean isParameter, ArrayList<Integer> shape, boolean onStack) {
        super(identifier, typeName, isParameter);
        this.shape = shape;
        this.onStack = onStack;
    }

    public ArrayData(String identifier, PrimitiveDataTypes typeName, boolean isInitialized, boolean isReturnData, ArrayList<Integer> shape, boolean onStack) {
        super(identifier, typeName, isInitialized, isReturnData);
        this.shape = shape;
        this.onStack = onStack;
    }

    public boolean isLocalPointer() {
        return isLocalPointer;
    }

    public void setLocalPointer(boolean localPointer) {
        isLocalPointer = localPointer;
    }

    /**
     * The size in bytes of the data type.
     * @return
     */
    @Override
    public int getSize() {
        int result = this.getTypeName().GetPrimitiveSize(this.getTypeName());
        for (int dimension: shape) {
            result *= dimension;
        }
        return result;
    }

    @Override
    public long getBytes() {
        long result = PrimitiveDataTypes.GetPrimitiveSize(this.getTypeName());
        for (Integer dim : shape) {
            result *= dim;
        }
        return result;
    }

    @Override
    public Data createInlineCopy(String inlineIdentifier) {
        return new ArrayData(this.getIdentifier() + "_" + inlineIdentifier, getTypeName(), isInitialized(), isReturnData(), new ArrayList<>(getShape()), isOnStack());
    }

    /**
     * Returns the dimensions of the multidimensional array as a list of integers, like the shape function in numpy.
     * @return
     */
    public ArrayList<Integer> getShape() {
        return shape;
    }

    public void setShape(ArrayList<Integer> shape) {
        this.shape = shape;
    }

    public boolean isOnStack() {
        return onStack;
    }

    @Override
    public ArrayData deepCopy() {
        return new ArrayData(getIdentifier(), getTypeName(), isInitialized(), isReturnData(), getShape(), onStack);
    }
}
