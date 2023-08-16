package de.parallelpatterndsl.patterndsl.generator;

import de.monticore.generating.GeneratorEngine;
import de.monticore.generating.GeneratorSetup;
import de.monticore.generating.templateengine.GlobalExtensionManagement;
import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;

import java.nio.file.Paths;

/**
 * Class that defines the source code generator for C++.
 */
public class PatternDSLGenerator {

    /**
     * The abstract pattern tree to generate the C++ sources from.
     */
    private AbstractMappingTree AMT;

    /**
     * The name of the output file.
     */
    private String filename;

    /**
     * Definition of the target network.
     */
    private Network network;

    /**
     * Path to the output files.
     */
    private String path;
    private String gpuThread;

    public PatternDSLGenerator(AbstractMappingTree AMT, String path ,String filename, Network network, String gpuThread) {
        this.AMT = AMT;
        this.filename = filename;
        this.network = network;
        this.path = path;
        this.gpuThread = gpuThread;
    }

    /**
     * The function generating the source files.
     */
    public void generate() {
        //Instantiate the helper class
        PatternDSLGeneratorHelper pfHelper = new PatternDSLGeneratorHelper(AMT,filename,network,gpuThread);
        GeneratorSetup gs = new GeneratorSetup();

        //define comment structure
        gs.setCommentStart("/*");
        gs.setCommentEnd("*/");
        gs.setTracing(false);

        //define output file ending
        gs.setDefaultFileExtension("cxx");

        //define the generator for the templates
        GlobalExtensionManagement management = new GlobalExtensionManagement();
        management.setGlobalValue("pfHelper", pfHelper);

        gs.setGlex(management);

        //generate the main source file
        GeneratorEngine engine = new GeneratorEngine(gs);
        engine.generateNoA("module.ftl", Paths.get(path,filename + ".cxx"), AMT);


        //Generate include
        gs.setDefaultFileExtension("hxx");
        engine = new GeneratorEngine(gs);
        //Function includes
        String name = filename.replaceAll(".cxx", "");
        engine.generateNoA("Functionheader.ftl", Paths.get(path,"includes",name + ".hxx"),AMT);
        engine.generateNoA("CudaPoolLibHeader.ftl", Paths.get(path,"includes","cuda_pool_lib" + ".hxx"));
        engine.generateNoA("PThreadsLibHeader.ftl", Paths.get(path,"includes","PThreadsLib.hxx"));
        engine.generateNoA("CudaLibHeader.ftl", Paths.get(path,"includes","cuda_lib_" + filename + ".hxx"),AMT);
        engine.generateNoA("LibraryHeader.ftl", Paths.get(path,"includes","Patternlib.hxx"),AMT);
        engine.generateNoA("TaskHeader.ftl", Paths.get(path,"includes","Task.hxx"),AMT);
        engine.generateNoA("TaskQueueHeader.ftl", Paths.get(path,"includes","TaskQueue.hxx"),AMT);
        engine.generateNoA("BitMaskHeader.ftl", Paths.get(path,"includes","BitMask.hxx"),AMT);

        //Generate include
        gs.setDefaultFileExtension("cxx");
        engine = new GeneratorEngine(gs);
        //Function includes
        engine.generateNoA("Library.ftl", Paths.get(path,"includes","Patternlib.cxx"),AMT);
        engine.generateNoA("Task.ftl", Paths.get(path,"includes","Task.cxx"),AMT);
        engine.generateNoA("BitMask.ftl", Paths.get(path,"includes","BitMask.cxx"),AMT);

        //PThreads includes
        engine.generateNoA("PThreadsLib.ftl", Paths.get(path,"includes","PThreadsLib.cxx"));
        engine.generateNoA("CudaPoolLib.ftl", Paths.get(path,"includes","cuda_pool_lib.cxx"));

        gs.setDefaultFileExtension("cu");
        engine = new GeneratorEngine(gs);
        //Function includes
        engine.generateNoA("CudaLib.ftl", Paths.get(path,"includes","cuda_lib_" + filename + ".cu"),AMT);

        gs.setDefaultFileExtension("cuh");
        engine = new GeneratorEngine(gs);
        engine.generateNoA("CudaLibKernelHeader.ftl", Paths.get(path,"includes","cuda_lib_" + filename + ".cuh"),AMT);


        gs.setDefaultFileExtension("");
        engine = new GeneratorEngine(gs);
        engine.generateNoA("Makefile.ftl", Paths.get(path,"Makefile"));
        engine.generateNoA("MachineFile.ftl", Paths.get(path,"MachineFile"));

    }
}
