# Project Execution on CÉCI Lyra

We highly recommend using a supercomputer for running training and post-processing, as these tasks are very heavy.

For our experiments, we used the **CÉCI Lyra ULB supercomputer** to run the experiments and fine-tune the model.

Official Lyra documentation:

- CÉCI Lyra page: <https://www.ceci-hpc.be/clusters/lyra/>
- ULB Lyra technical page: <https://hpc.ulb.be/lyra.php>
- CÉCI documentation: <https://support.ceci-hpc.be/doc/>
- Slurm job submission tutorial: <https://support.ceci-hpc.be/doc/SubmittingJobs/SlurmTutorial/>

In addition to the official documentation, we provide below some tips and tricks that were useful for this project.

---

## 1. SSH Setup with VS Code

We recommend setting up Lyra through an SSH connection in VS Code. This is much easier to handle, especially with an SSH configuration that helps avoid Lyra timeouts.

Add the following configuration to your SSH config file (replace USERNAME with your username:

```sshconfig

Host gwceci
    HostName gwceci.cism.ucl.ac.be
    User USERNAME
    IdentityFile ~/.ssh/id_rsa.ceci
    
    
Host lyra lemaitre4 hercules nic5 dragon2 manneback
    User USERNAME
    ForwardX11 yes
    IdentityFile ~/.ssh/id_rsa.ceci
    ProxyJump gwceci
    ServerAliveInterval 120
    ServerAliveCountMax 100

Host lyra
    HostName lyra.ulb.be
Host lemaitre4
    HostName lemaitre4.cism.ucl.ac.be
Host hercules
    HostName hercules.ptci.unamur.be
Host dragon2
    HostName dragon2.umons.ac.be
Host nic5
    HostName nic5.uliege.be
Host manneback
    HostName manneback.cism.ucl.ac.be
```

---

## 2. Repository and Storage Organization

The GitHub repository can be cloned in the **Home directory**.

However, do **not** download all images directly inside the Home directory, because the Home directory is limited to **100 GB per user**. It also has a maximum number of files, which this project can easily exceed.

Instead, download the images and large files inside the global working directory:


```bash
cd $GLOBALSCRATCH
```

This directory provides **5 TB per user** and has a much higher capacity.

```bash
/globalsc/ucl/ingi/<username>/ChimpRec/
├── ChimpPic
├── ChimpVideos        
├── .venv          
```

```bash
~/ChimpRec       (GitHub repository)
├── ChimpPic     (symlink)
├── ChimpVideos  (symlink)       
├── .venv        (symlink)
├── Code        
├── Models        
```

---

## 3. Python Virtual Environment

To avoid reaching the file number limit in the Home directory, place the Python virtual environment inside the global directory as well.

This is important because the project requires many Python libraries.

---

## 4. Links Inside the `ChimpRec` Repository

Place links inside the `ChimpRec` repository at the same level as the `Code` directory.

The large files should remain in the global directory, while the repository can access them through these links.

---

## 5. Downloading Videos from OneDrive

To download videos from OneDrive, we recommend using the `rclone` library.

Download the videos directly from OneDrive to Lyra using Lyra's high-speed connection, around **50–100 Mb/s**.

This is far more efficient than first downloading the videos to your computer and then uploading them to Lyra.

---

## Summary

- Use a supercomputer for training and post-processing.
- We used CÉCI Lyra ULB for the experiments and model fine-tuning.
- Use VS Code with SSH to connect to Lyra more easily.
- Clone the GitHub repository in the Home directory.
- Do not store all images in the Home directory.
- Store images, large files, and the Python virtual environment in `/globalsc` or `$GLOBALSCRATCH`.
- Place links inside the `chimpRec` repository at the same level as the `Code` directory.
- Use `rclone` to download videos directly from OneDrive to Lyra.
