# Lightning SSH Setup

This guide records the SSH setup flow that worked for the Lightning Studio machines.

## What This Solves

- new machine setup
- re-enabling SSH after a machine reset or sleep
- Windows SSH configuration for Lightning
- verifying that the local key is actually accepted by the remote Studio

## The Setup Flow

1. Open the Lightning SSH setup page for the target machine.
2. Run the Windows setup script in PowerShell.
3. Confirm that the keypair is created in `~/.ssh/`.
4. Confirm the SSH config entry points at the new private key.
5. Test the connection with a one-shot SSH command.

## What The Script Does

The Lightning setup script creates:

- `~/.ssh/lightning_rsa`
- `~/.ssh/lightning_rsa.pub`
- an SSH config entry for `ssh.lightning.ai`

The config usually looks like this:

```sshconfig
Host ssh.lightning.ai
  IdentityFile C:\Users\<you>\.ssh\lightning_rsa
  IdentitiesOnly yes
  ServerAliveInterval 15
  ServerAliveCountMax 4
  StrictHostKeyChecking no
  UserKnownHostsFile=\\.\NUL
```

## Verification Commands

Check that the private key exists and can generate a public key:

```bash
ssh-keygen -y -f $env:USERPROFILE\.ssh\lightning_rsa
```

Test the login with a one-shot command:

```bash
ssh -o BatchMode=yes s_01kt2crz4evvhcs9x7gw01adtv@ssh.lightning.ai "echo ok"
```

If that prints `ok`, SSH is working.

## If You Get `Permission denied (publickey)`

That usually means the local key exists, but the Studio does not currently accept it.

Use Lightning's UI and re-run the **setup new computer** flow for SSH access, then retry the login test.

## Windows Note

The Lightning script may fail on the `Set-Acl` step if PowerShell does not have the required privilege.
If that happens, the key files may still exist and still be usable.

The important part is the remote login test:

- if it succeeds, the key is registered correctly
- if it fails, the Studio needs the SSH key re-authorized

## What Not To Do

- Do not put GitHub, Hugging Face, or Docker credentials into repo files.
- Do not keep a long training session on a local desktop machine.
- Do not assume a key that worked on one Studio will work forever on a new Studio.

## Related Docs

- [Start Here](../../START_HERE.md)
- [Machine switch quick reference](../../MACHINE_SWITCH_QUICK_REF.md)
- [Remote workflow](../../REMOTE_WORKFLOW.md)

