package de.parallelpatterndsl.patterndsl.MappingTree.GeneralDataPlacementFunctions;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.EndPoint;
import de.parallelpatterndsl.patterndsl.MappingTree.SupportFunction;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;
import de.se_rwth.commons.logging.Log;

public class OffloadDataEncoding {

    private String identifier;

    private String baseIdentifier;

    private DataPlacement data;

    private String originalIdentifier;

    private boolean markedForDeallocation;

    private boolean writeAccessed;

    private long startHalo;

    private boolean reductionResult;

    public OffloadDataEncoding(DataPlacement data, boolean writeAccessed, long startHalo) {
        this.data = data;
        this.startHalo = startHalo;
        identifier = "GPU_Data_" + RandomStringGenerator.getAlphaNumericString();
        this.writeAccessed = writeAccessed;
        markedForDeallocation = false;
        originalIdentifier = data.getDataElement().getIdentifier();
        baseIdentifier = data.getDataElement().getBaseIdentifier();
        reductionResult = false;
    }

    public String getIdentifier() {
        return identifier;
    }

    public DataPlacement getDataPlacement() {
        return data;
    }

    public Data getData() {
        return data.getDataElement();
    }

    public boolean isReductionResult() {
        return reductionResult;
    }

    public void setReductionResult(boolean reductionResult) {
        this.reductionResult = reductionResult;
    }

    public Device getDevice() {
        if (data.getPlacement().size() != 1) {
            Log.error("Offloading data encoding contained more than one EndPoint or no EndPoint for variable: " + data.getDataElement().getIdentifier());
            System.exit(1);
        }
        return data.getPlacement().get(0).getLocation();
    }

    public long getStart() {
        if (data.getPlacement().size() != 1) {
            Log.error("Offloading data encoding contained more than one EndPoint or no EndPoint for variable: " + data.getDataElement().getIdentifier());
            System.exit(1);
        }
        long start = data.getPlacement().get(0).getStart();
        return start;
    }

    public long getRealStart() {
        if (data.getPlacement().size() != 1) {
            Log.error("Offloading data encoding contained more than one EndPoint or no EndPoint for variable: " + data.getDataElement().getIdentifier());
            System.exit(1);
        }
        long start = data.getPlacement().get(0).getStart();
        if (data.getDataElement() instanceof ArrayData) {
            start = Long.min(start, ((ArrayData) data.getDataElement()).getShape().get(0));
            for (int i = 1; i < ((ArrayData) data.getDataElement()).getShape().size(); i++) {
                start *= ((ArrayData) data.getDataElement()).getShape().get(i);
            }
        }
        return start;
    }

    public long getRealStartHalo() {
        long realHalo = startHalo;
        if (data.getDataElement() instanceof ArrayData) {
            for (int i = 1; i < ((ArrayData) data.getDataElement()).getShape().size(); i++) {
                realHalo *= ((ArrayData) data.getDataElement()).getShape().get(i);
            }
        }
        return realHalo;
    }

    public long getLength() {
        if (data.getPlacement().size() != 1) {
            Log.error("Offloading data encoding contained more than one EndPoint or no EndPoint for variable: " + data.getDataElement().getIdentifier());
            System.exit(1);
        }
        long printLength = data.getPlacement().get(0).getLength() ;

        if (data.getDataElement() instanceof ArrayData) {
            printLength = Long.min(printLength, ((ArrayData) data.getDataElement()).getShape().get(0));
            for (int i = 1; i < ((ArrayData) data.getDataElement()).getShape().size(); i++) {
                printLength *= ((ArrayData) data.getDataElement()).getShape().get(i);
            }
        }

        return printLength;
    }

    public void updateDataIdentifier() {
        if (getData() instanceof PrimitiveData) {
            getData().setIdentifier(getData().getIdentifier() + "[0]");
        }
    }

    public void resetDataIdentifier() {
        if (getData() instanceof PrimitiveData) {
            int inlinerLength = 0;
            if (getData().hasInlineIdentifier()) {
                inlinerLength = 1 + getData().getInlineIdentifier().length();
            }
            getData().setIdentifier(getData().getIdentifier().substring(0, getData().getIdentifier().length() - (3 + inlinerLength)));
        }
    }

    public void updateDataIdentifier(boolean reduction) {
        if (reduction) {
            if (getData() instanceof PrimitiveData) {
                getData().setIdentifier(getData().getIdentifier() + "_s[id]");
            }
        }
    }

    public void resetDataIdentifier(boolean reduction) {
        if (reduction) {
            if (getData() instanceof PrimitiveData) {
                getData().setIdentifier(baseIdentifier);
            }
        }
    }

    public boolean isMarkedForDeallocation() {
        return markedForDeallocation;
    }

    public void setMarkedForDeallocation(boolean markedForDeallocation) {
        this.markedForDeallocation = markedForDeallocation;
    }

    public boolean isWriteAccessed() {
        return writeAccessed;
    }

    public void setWriteAccessed(boolean writeAccessed) {
        this.writeAccessed = writeAccessed;
    }
}
