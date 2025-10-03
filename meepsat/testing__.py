def at_every_command(at_every_list, timestep):
    if not at_every_list:  # Handle empty list case
        return ""
    
    command = ""
    for data in at_every_list:
        command += f"mp.at_every({timestep}, mp_sim_run.{data}),\n"
    
    return command.rstrip(",\n")  # Remove trailing comma and newline if present

def at_end_command(at_end_list):
    if not at_end_list:  # Handle empty list case
        return ""
        
    command = ""
    for data in at_end_list:
        command += f"mp.at_end(mp.{data}),\n"
    
    return command.rstrip(",\n")  # Remove trailing comma and newline if present

def data_required_script_inside_sim_run(data_required):
    script = ""
    script += at_every_command(data_required["at_every"], "image_every")
    # Add a comma and newline 
    script += ",\n"
    script += at_end_command(data_required["at_end"])
    return script

def sims_data_requested(data_required, run_time):
    # Prepare the content for the run function separately
    run_args = data_required_script_inside_sim_run(data_required)
    # If run_args is not empty, add a comma between commands
    if run_args:
        run_args += ",\n"
    run_args += run_time["command"]
    
    # Apply indentation to each line of run_args
    indented_args = "\n".join(f"    {line}" for line in run_args.split("\n"))
    
    main_script = f"""self.sim.run(
{indented_args}
)"""
    print(main_script)
    return main_script

##############################
if __name__ == "__main__":
    # Just directly add the until or until_after_sources commands
    runtime = {
        "command": "until= 2000"
    }
    
    data_required= {
            "at_every_timestep": 5,
            "at_every": ["animate_Ez2_dB", "Ez2_dB"],
            "at_end": ["output_efield_z"]
            }

    sims_data_requested(data_required, runtime)
