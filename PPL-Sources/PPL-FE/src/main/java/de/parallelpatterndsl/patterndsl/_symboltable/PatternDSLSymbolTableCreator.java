package de.parallelpatterndsl.patterndsl._symboltable;

import de.monticore.symboltable.ArtifactScope;
import de.monticore.symboltable.ResolvingConfiguration;
import de.monticore.symboltable.Scope;
import de.parallelpatterndsl.patterndsl._ast.*;

import java.util.ArrayList;
import java.util.Deque;
import java.util.Optional;

import static java.util.Objects.requireNonNull;

/**
 * Class that extends the creation of the symbol table, to incorporate the changes to the symbols.
 */
public class PatternDSLSymbolTableCreator extends PatternDSLSymbolTableCreatorTOP {
    protected ASTModule module;

    boolean global = true;

    public PatternDSLSymbolTableCreator(ResolvingConfiguration resolvingConfig, Scope enclosingScope) {
        super(resolvingConfig, enclosingScope);
    }

    public PatternDSLSymbolTableCreator(ResolvingConfiguration resolvingConfig, Deque<Scope> scopeStack) {
        super(resolvingConfig, scopeStack);
    }

    /**
     * Starts the creation of the symbol table
     *
     * @param rootNode
     * @return
     */
    @Override
    public Scope createFromAST(ASTModule rootNode) {
        requireNonNull(rootNode);
        module = rootNode;

        final ArtifactScope artifactScope = new ArtifactScope(Optional.empty(), "", new ArrayList<>());
        putOnStack(artifactScope);
        rootNode.accept(this);

        return artifactScope;
    }

    /**
     * Override the creation of a variable Symbol and extend it with
     * additional information for the symbol
     * e.g. which type it has
     *
     *
     * @param ast
     * @return
     */
    @Override
    protected VariableSymbol create_Variable(ASTVariable ast) {
        boolean arrayOnStack = false;
        if (ast.isPresentExpression()) {
            if (ast.getExpression() instanceof ASTListExpression) {
                arrayOnStack = true;
            }
        }
        return new VariableSymbol(ast.getName(), ast.getType(), global, arrayOnStack);
    }

    @Override
    public void traverse(ASTFunction node) {
        // One might think that we could call traverse(subelement) immediately,
        // but this is not true for interface-types where we do not know the
        // concrete type of the element.
        // Instead we double-dispatch the call, to call the correctly typed
        // traverse(...) method with the elements concrete type.
        global = false;
        if (null != node.getPatternType()) {
            node.getPatternType().accept(getRealThis());
        }
        if (null != node.getFunctionParameters()) {
            node.getFunctionParameters().accept(getRealThis());
        }
        if (node.getTypeOpt().isPresent()) {
            node.getTypeOpt().get().accept(getRealThis());
        }
        if (node.getFunctionParameterOpt().isPresent()) {
            node.getFunctionParameterOpt().get().accept(getRealThis());
        }
        if (null != node.getBlockStatement()) {
            node.getBlockStatement().accept(getRealThis());
        }
        global=true;
    }

    /**
     * Override the creation of a function parameter Symbol and extend it with
     * additional information for the symbol
     * e.g. which type it has
     *
     * @param ast
     * @return
     */
    @Override
    protected FunctionParameterSymbol create_FunctionParameter(ASTFunctionParameter ast) {
        return new FunctionParameterSymbol(ast.getName(), ast.getType());
    }

    /**
     * Override the creation of a function Symbol and extend it with
     * additional information for the symbol
     * e.g. which pattern type it has
     *
     * @param ast
     * @return
     */
    @Override
    protected FunctionSymbol create_Function(ASTFunction ast) {
        return new FunctionSymbol(ast.getName(), ast.getPatternType(), ast.getFunctionParameters().sizeFunctionParameters());
    }

    /**
     * Override the creation of a constant Symbol and extend it with
     * additional information for the symbol
     * e.g. which type it has
     *
     * @param ast
     * @return
     */
    @Override
    protected ConstantSymbol create_Constant(ASTConstant ast) {
        return new ConstantSymbol(ast.getName(),ast.getListExpressionOpt(),ast.getNameExpressionOpt(),ast.getLiteralExpressionOpt());
    }
}
