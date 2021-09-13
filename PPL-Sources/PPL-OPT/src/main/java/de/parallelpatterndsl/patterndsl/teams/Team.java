package de.parallelpatterndsl.patterndsl.teams;

import com.google.gson.annotations.Expose;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;

/**
 * The Team class abstracts the largest group of UEs on a device sharing a common cache, e.g., L3 cache on CPU, SM on a GPU.
 */
public class Team implements Cloneable {

    private final Device device;

    private final Processor processor;

    private int cores;

    public Team(Device device, Processor processor, int cores) {
        this.device = device;
        this.processor = processor;
        this.cores = cores;
    }

    public Team(Team team) {
        this.device = team.device;
        this.processor = team.processor;
        this.cores = team.cores;
    }

    /**
     * Returns the associated device.
     * @return Device.
     */
    public Device getDevice() {
        return device;
    }

    /**
     * Returns the associated processor.
     * @return Processor.
     */
    public Processor getProcessor() {
        return processor;
    }

    /**
     * Returns the number of occupied cores.
     * @return int.
     */
    public int getCores() {
        return cores;
    }

    /**
     * Sets the number of occupied cores.
     * @param cores - cores to be occupied.
     */
    public void setCores(int cores) {
        this.cores = cores;
    }

}
