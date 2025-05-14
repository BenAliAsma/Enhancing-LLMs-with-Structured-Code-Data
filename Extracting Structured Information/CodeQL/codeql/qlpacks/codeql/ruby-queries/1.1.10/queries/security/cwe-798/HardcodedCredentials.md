# Hard-coded credentials
Including unencrypted hard-coded inbound or outbound authentication credentials within source code or configuration files is dangerous because the credentials may be easily discovered.

Source or configuration files containing hard-coded credentials may be visible to an attacker. For example, the source code may be open source, or it may be leaked or accidentally revealed.

For inbound authentication, hard-coded credentials may allow unauthorized access to the system. This is particularly problematic if the credential is hard-coded in the source code, because it cannot be disabled easily. For outbound authentication, the hard-coded credentials may provide an attacker with privileged information or unauthorized access to some other system.


## Recommendation
Remove hard-coded credentials, such as user names, passwords and certificates, from source code, placing them in configuration files or other data stores if necessary. If possible, store configuration files including credential data separately from the source code, in a secure location with restricted access.

For outbound authentication details, consider encrypting the credentials or the enclosing data stores or configuration files, and using permissions to restrict access.

For inbound authentication details, consider hashing passwords using standard library functions where possible. For example, `OpenSSL::KDF.pbkdf2_hmac`.


## Example
The following examples shows different types of inbound and outbound authentication.

In the first case, `RackAppBad`, we accept a password from a remote user, and compare it against a plaintext string literal. If an attacker acquires the source code they can observe the password, and can log in to the system. Furthermore, if such an intrusion was discovered, the application would need to be rewritten and redeployed in order to change the password.

In the second case, `RackAppGood`, the password is compared to a hashed and salted password stored in a configuration file, using `OpenSSL::KDF.pbkdf2_hmac`. In this case, access to the source code or the assembly would not reveal the password to an attacker. Even access to the configuration file containing the password hash and salt would be of little value to an attacker, as it is usually extremely difficult to reverse engineer the password from the hash and salt. In a real application care should be taken to make the string comparison of the hashed input against the hashed password take close to constant time, as this will make timing attacks more difficult.


```ruby
require 'rack'
require 'yaml'
require 'openssl'

class RackAppBad
  def call(env)
    req = Rack::Request.new(env)
    password = req.params['password']

    # BAD: Inbound authentication made by comparison to string literal
    if password == 'myPa55word'
      [200, {'Content-type' => 'text/plain'}, ['OK']]
    else
      [403, {'Content-type' => 'text/plain'}, ['Permission denied']]
    end
  end
end

class RackAppGood
  def call(env)
    req = Rack::Request.new(env)
    password = req.params['password']

    config_file = YAML.load_file('config.yml')
    hashed_password = config_file['hashed_password']
    salt = [config_file['salt']].pack('H*')

    #GOOD: Inbound authentication made by comparing to a hash password from a config file.
    hash = OpenSSL::Digest::SHA256.new
    dk = OpenSSL::KDF.pbkdf2_hmac(
      password, salt: salt, hash: hash, iterations: 100_000, length: hash.digest_length
    )
    hashed_input = dk.unpack('H*').first
    if hashed_password == hashed_input
      [200, {'Content-type' => 'text/plain'}, ['OK']]
    else
      [403, {'Content-type' => 'text/plain'}, ['Permission denied']]
    end
  end
end

```

## References
* OWASP: [XSS Use of hard-coded password](https://www.owasp.org/index.php/Use_of_hard-coded_password).
* Common Weakness Enumeration: [CWE-259](https://cwe.mitre.org/data/definitions/259.html).
* Common Weakness Enumeration: [CWE-321](https://cwe.mitre.org/data/definitions/321.html).
* Common Weakness Enumeration: [CWE-798](https://cwe.mitre.org/data/definitions/798.html).
