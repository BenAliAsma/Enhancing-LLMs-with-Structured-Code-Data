# Missing read or write permission in a content provider
The Android manifest file specifies the content providers for the application using `provider` elements. The `provider` element specifies the explicit permissions an application requires in order to access a resource using that provider. You specify the permissions using the `android:readPermission`, `android:writePermission`, or `android:permission` attributes. If you do not specify the permission required to perform an operation, the application will implicitly have access to perform that operation. For example, if you specify only `android:readPermission`, the application must have explicit permission to read data, but requires no permission to write data.


## Recommendation
To prevent permission bypass, you should create `provider` elements that either specify both the `android:readPermission` and `android:writePermission` attributes, or specify the `android:permission` attribute.


## Example
In the following two (bad) examples, the provider is configured with only read or write permissions. This allows a malicious application to bypass the permission check by requesting access to the unrestricted operation.


```xml
<manifest ... >
    <application ...>
      <!-- BAD: only 'android:readPermission' is set -->
      <provider
          android:name=".MyContentProvider"
          android:authorities="table"
          android:enabled="true"
          android:exported="true"
          android:readPermission="android.permission.MANAGE_DOCUMENTS">
      </provider>
    </application>
</manifest>

```

```xml
<manifest ... >
    <application ...>
      <!-- BAD: only 'android:writePermission' is set -->
      <provider
          android:name=".MyContentProvider"
          android:authorities="table"
          android:enabled="true"
          android:exported="true"
          android:writePermission="android.permission.MANAGE_DOCUMENTS">
      </provider>
    </application>
</manifest>

```
In the following (good) examples, the provider is configured with full permissions, protecting it from a permissions bypass.


```xml
<manifest ... >
    <application ...>
      <!-- Good: both 'android:readPermission' and 'android:writePermission' are set -->
      <provider
          android:name=".MyContentProvider"
          android:authorities="table"
          android:enabled="true"
          android:exported="true"
          android:writePermission="android.permission.MANAGE_DOCUMENTS"
          android:readPermission="android.permission.MANAGE_DOCUMENTS">
      </provider>
    </application>
</manifest>

```

```xml
<manifest ... >
    <application ...>
      <!-- Good: 'android:permission' is set  -->
      <provider
          android:name=".MyContentProvider"
          android:authorities="table"
          android:enabled="true"
          android:exported="true"
          android:permission="android.permission.MANAGE_DOCUMENTS">
      </provider>
    </application>
</manifest>

```

## References
* Android Documentation: [Provider element](https://developer.android.com/guide/topics/manifest/provider-element)
* CVE-2021-41166: [Insufficient permission control in Nextcloud Android app](https://nvd.nist.gov/vuln/detail/CVE-2021-41166)
* GitHub Security Lab Research: [Insufficient permission control in Nextcloud Android app](https://securitylab.github.com/advisories/GHSL-2021-1007-Nextcloud_Android_app/#issue-2-permission-bypass-in-disklruimagecachefileprovider-ghsl-2021-1008)
* Common Weakness Enumeration: [CWE-926](https://cwe.mitre.org/data/definitions/926.html).
