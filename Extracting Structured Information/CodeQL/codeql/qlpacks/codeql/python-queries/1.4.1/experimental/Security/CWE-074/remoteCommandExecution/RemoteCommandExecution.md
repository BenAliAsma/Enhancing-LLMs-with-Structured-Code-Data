# Command execution on a secondary remote server
Allowing users to execute arbitrary commands using an SSH connection on a remote server can lead to security issues unless you implement proper authorization.

Assume that you connect to a remote system via SSH connection from your main or local server that accepts user-controlled data and has interaction with users that you don't trust, passing these data to SSH API as a part of a command that will be executed on a secondary remote server can lead to security issues. You should consider proper authorization rules very carefully.


## Recommendation
This vulnerability can be prevented by implementing proper authorization rules for untrusted user input that can be passed to your secondary servers.


## Example
In the example below, the exec_command is controlled by the user and hence leads to a vulnerability.


```python
#!/usr/bin/env python

from flask import request, Flask
import paramiko
from paramiko import SSHClient

app = Flask(__name__)
paramiko_ssh_client = SSHClient()
paramiko_ssh_client.load_system_host_keys()
paramiko_ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
paramiko_ssh_client.connect(hostname="127.0.0.1", port="22", username="ssh_user_name", pkey="k", timeout=11, banner_timeout=200)


@app.route('/external_exec_command_1')
def withoutAuthorization():
    user_cmd = request.args.get('command')
    stdin, stdout, stderr = paramiko_ssh_client.exec_command(user_cmd)
    return stdout

if __name__ == '__main__':
    app.debug = False
    app.run()


```
In the example below, the exec_command is controlled by the an Authorized user and hence it is safe.


```python
#!/usr/bin/env python

from flask import request, Flask
import paramiko
from paramiko import SSHClient

app = Flask(__name__)
paramiko_ssh_client = SSHClient()
paramiko_ssh_client.load_system_host_keys()
paramiko_ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
paramiko_ssh_client.connect(hostname="127.0.0.1", port="22", username="ssh_user_name", pkey="k", timeout=11, banner_timeout=200)


@app.route('/external_exec_command_1')
def withAuthorization():
    user_cmd = request.args.get('command')
    auth_jwt = request.args.get('Auth')
    # validating jwt token first
    # .... then continue to run the command
    stdin, stdout, stderr = paramiko_ssh_client.exec_command(user_cmd)
    return stdout


if __name__ == '__main__':
    app.debug = False
    app.run()


```
