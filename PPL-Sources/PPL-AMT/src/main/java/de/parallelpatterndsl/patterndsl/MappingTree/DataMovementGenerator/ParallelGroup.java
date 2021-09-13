package de.parallelpatterndsl.patterndsl.MappingTree.DataMovementGenerator;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.MainMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.ParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain.SimpleExpressionBlockMapping;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Optional;
import java.util.Set;

public abstract class ParallelGroup {

    public abstract Optional<SimpleExpressionBlockMapping> getParameterReplacementExpressions();

    public abstract Optional<SimpleExpressionBlockMapping> getResultReplacementExpression();

    public abstract boolean isFirstAccess();

    public abstract void setFirstAccess(boolean firstAccess);

    public abstract MainMapping getMainFunction();

    public abstract ArrayList getGroup();

    public abstract int getRemaining();

    public abstract boolean isLastCall();

    public abstract HashMap<Data,ArrayList<DataPlacement>> getFullInputPlacement();

    public abstract void resetRemaining();

    public abstract String getGroupIdentifier();

    public abstract Set<Processor> getProcessors();
}
