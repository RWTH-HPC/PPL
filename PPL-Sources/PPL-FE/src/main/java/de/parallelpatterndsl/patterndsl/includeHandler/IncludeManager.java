package de.parallelpatterndsl.patterndsl.includeHandler;

import de.monticore.symboltable.GlobalScope;
import de.parallelpatterndsl.patterndsl._ast.ASTDefinition;
import de.parallelpatterndsl.patterndsl._ast.ASTModule;
import de.parallelpatterndsl.patterndsl._symboltable.ModuleSymbol;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;

/**
 * This class defines how to handle the inclusion of additional modules by extending the symbol-table and AST of the main source file.
 */
public abstract class IncludeManager {


    protected GlobalScope symbolTable;

    protected ModuleSymbol mainSource;

    protected ASTModule mainModule;

    protected String sourcePath;

    protected ArrayList<String> includedModules = new ArrayList<>();

    public IncludeManager(GlobalScope symbolTable, ModuleSymbol mainSource, String sourcePath) {
        this.symbolTable = symbolTable;
        this.mainSource = mainSource;
        this.sourcePath = sourcePath;

        if (mainSource.getModuleNode().isPresent()) {
            mainModule = mainSource.getModuleNode().get();
        } else {
            Log.error("Main module not pressent!");
        }
    }

    /**
     * This function handles all includes of the main file and extends it by all included contents
     */
    abstract public void handleIncludes();

    /**
     * Handles a single included module
     * @param modulePath
     */
    abstract public void include2Main(String modulePath);

    /**
     * Handles the extension of the AST for a single element of one included module defined by its index
     * @param includedModule
     * @param extensionIndex
     */
    abstract public void extendModule(ASTModule includedModule, int extensionIndex);

    /**
     * Handles the extension of the symbol table for a single element of one included module defined by its index
     * @param includedSymbolTable
     * @param name
     */
    abstract public void extendSymbolTable(ModuleSymbol includedSymbolTable, String name);
}
