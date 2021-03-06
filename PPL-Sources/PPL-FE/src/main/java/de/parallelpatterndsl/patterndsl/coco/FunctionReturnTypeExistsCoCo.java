package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, that for sequential functions only the return type is defined.
 */
public class FunctionReturnTypeExistsCoCo implements PatternDSLASTFunctionCoCo {

    @Override
    public void check(ASTFunction node) {
        if (node.isPresentType() || !node.getPatternType().isPresentSerial()) {
            return;
        }
        Log.error(node.get_SourcePositionStart() + " No return type defined or redundant return variable defined: " + node.getName());
    }

}
