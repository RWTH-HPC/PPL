package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl.coco.Helper.ReadOnlyHelperConstants;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, if constants are only read from.
 */
public class ReadOnlyConstantsCoCo implements PatternDSLASTFunctionCoCo {

    @Override
    public void check(ASTFunction node) {
        ReadOnlyHelperConstants helper = new ReadOnlyHelperConstants();
        if (!helper.isReadOnly(node)) {
            Log.error(node.get_SourcePositionStart() + " Change of constant detected");
        }
    }


}
