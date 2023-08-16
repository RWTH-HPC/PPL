package de.parallelpatterndsl.patterndsl.printer;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.GPUAllocationMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.GPUDataMovementMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.GPUDeAllocationMapping;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;

public class CppGPUDataNodePrinter {

    public static void generateGPUAllocMapping(GPUAllocationMapping node, StringBuilder builder) {
        builder.append("cuda_alloc_wrapper(&");
        builder.append(node.getAllocator().getIdentifier());
        builder.append(", sizeof(");
        builder.append(CppTypesPrinter.doPrintType(node.getAllocator().getData().getTypeName()));
        builder.append(") * ");
        builder.append(node.getAllocator().getLength() + node.getAllocator().getRealStartHalo());
        builder.append(");\n");
    }

    public static void generateGPUDeAllocMapping(GPUDeAllocationMapping node, StringBuilder builder) {
        builder.append("cuda_dealloc_wrapper(");
        builder.append(node.getAllocator().getIdentifier());
        builder.append(");\n");
    }

    public static void generateGPUDataMovementMapping(GPUDataMovementMapping node, StringBuilder builder) {
        if (node.isCPU2GPU()) {
            builder.append("cuda_host2device_wrapper(&");
            builder.append(node.getAllocator().getIdentifier());
            builder.append("[");
            builder.append(node.getAllocator().getRealStartHalo());
            builder.append("], ");
            builder.append("&");
            builder.append(node.getAllocator().getData().getIdentifier());
            if (node.getAllocator().getData() instanceof ArrayData) {
                builder.append("[");
                builder.append(node.getAllocator().getRealStart());
                builder.append("]");
            }
            builder.append(", sizeof(");
            builder.append(CppTypesPrinter.doPrintType(node.getAllocator().getData().getTypeName()));
            builder.append(") * ");
            builder.append(node.getAllocator().getLength());
            builder.append(");\n");
        } else {
            builder.append("cuda_device2host_wrapper(");
            builder.append("&");
            builder.append(node.getAllocator().getData().getIdentifier());
            if (node.getAllocator().getData() instanceof ArrayData) {
                builder.append("[");
                builder.append(node.getAllocator().getRealStart());
                builder.append("]");
            }
            builder.append(", &");
            builder.append(node.getAllocator().getIdentifier());
            builder.append("[");
            builder.append(node.getAllocator().getRealStartHalo());
            builder.append("], sizeof(");
            builder.append(CppTypesPrinter.doPrintType(node.getAllocator().getData().getTypeName()));
            builder.append(") * ");
            builder.append(node.getAllocator().getLength());
            builder.append(");\n");
        }
    }
}
