package de.parallelpatterndsl.patterndsl.coco;

import de.monticore.expressions.commonexpressions._ast.ASTCallExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTConstant;
import de.parallelpatterndsl.patterndsl._ast.ASTForEachControl;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTConstantCoCo;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTForEachControlCoCo;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.se_rwth.commons.logging.Log;

/**
 * Defines a CoCo that checks, that the variable in for each loops is not assigned.
 */
public class NoForEachLoopAssignmentCoCo implements PatternDSLASTForEachControlCoCo {

    @Override
    public void check(ASTForEachControl node) {
        if (node.getVariable().isPresentExpression()) {
            Log.error(node.get_SourcePositionStart() + " Do not assign a value in the for each loop control!");
        }
    }

}
