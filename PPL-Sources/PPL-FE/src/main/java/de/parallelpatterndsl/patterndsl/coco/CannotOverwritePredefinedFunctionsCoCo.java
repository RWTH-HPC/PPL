package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, if the return name is defined.
 */
public class CannotOverwritePredefinedFunctionsCoCo implements PatternDSLASTFunctionCoCo {

    @Override
    public void check(ASTFunction node) {
        if (PredefinedFunctions.contains(node.getName())) {
            Log.error(node.get_SourcePositionStart() + " You cannot overwrite predefined functions: " + node.getName());
        }

    }

}
