# Android APK installation
Android allows an application to install an Android Package Kit (APK) using an `Intent` with the `"application/vnd.android.package-archive"` MIME type. If the file used in the `Intent` is from a location that is not controlled by the application (for example, an SD card that is universally writable), this can result in the unintended installation of untrusted applications.


## Recommendation
You should install packages using the `PackageInstaller` class.

If you need to install from a file, you should use a `FileProvider`. Content providers can provide more specific permissions than file system permissions can.

When your application does not require package installations, do not add the `REQUEST_INSTALL_PACKAGES` permission in the manifest file.


## Example
In the following (bad) example, the package is installed from a file which may be altered by another application:


```java
import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Environment;

import java.io.File;

/* Get a file from external storage */
File file = new File(Environment.getExternalStorageDirectory(), "myapp.apk");
Intent intent = new Intent(Intent.ACTION_VIEW);
/* Set the mimetype to APK */
intent.setDataAndType(Uri.fromFile(file), "application/vnd.android.package-archive");

startActivity(intent);

```
In the following (good) example, the package is installed by using a `FileProvider`:


```java
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import androidx.core.content.FileProvider;

import java.io.File;
import java.io.FileOutputStream;

String tempFilename = "temporary.apk";
byte[] buffer = new byte[16384];

/* Copy application asset into temporary file */
try (InputStream is = getAssets().open(assetName);
     FileOutputStream fout = openFileOutput(tempFilename, Context.MODE_PRIVATE)) {
    int n;
    while ((n=is.read(buffer)) >= 0) {
        fout.write(buffer, 0, n);
    }
}

/* Expose temporary file with FileProvider */
File toInstall = new File(this.getFilesDir(), tempFilename);
Uri applicationUri = FileProvider.getUriForFile(this, "com.example.apkprovider", toInstall);

/* Create Intent and set data to APK file. */
Intent intent = new Intent(Intent.ACTION_INSTALL_PACKAGE);
intent.setData(applicationUri);
intent.setFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);

startActivity(intent);

```
In the following (good) example, the package is installed using an instance of the `android.content.pm.PackageInstaller` class:


```java
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInstaller;

private static final String PACKAGE_INSTALLED_ACTION =
    "com.example.SESSION_API_PACKAGE_INSTALLED";

/* Create the package installer and session */
PackageInstaller packageInstaller = getPackageManager().getPackageInstaller();
PackageInstaller.SessionParams params =
    new PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL);
int sessionId = packageInstaller.createSession(params);
session = packageInstaller.openSession(sessionId);

/* Load asset into session */
try (OutputStream packageInSession = session.openWrite("package", 0, -1);
     InputStream is = getAssets().open(assetName)) {
    byte[] buffer = new byte[16384];
    int n;
    while ((n = is.read(buffer)) >= 0) {
        packageInSession.write(buffer, 0, n);
    }
}

/* Create status receiver */
Intent intent = new Intent(this, InstallApkSessionApi.class);
intent.setAction(PACKAGE_INSTALLED_ACTION);
PendingIntent pendingIntent = PendingIntent.getActivity(context, 0, intent, 0);
IntentSender statusReceiver = pendingIntent.getIntentSender();

/* Commit the session */
session.commit(statusReceiver);

```

## References
* Android Developers: [Intent.ACTION_INSTALL_PACKAGE](https://developer.android.com/reference/android/content/Intent#ACTION_INSTALL_PACKAGE).
* Android Developers: [Manifest.permission.REQUEST_INSTALL_PACKAGES](https://developer.android.com/reference/android/Manifest.permission#REQUEST_INSTALL_PACKAGES).
* Android Developers: [PackageInstaller](https://developer.android.com/reference/android/content/pm/PackageInstaller).
* Android Developers: [FileProvider](https://developer.android.com/reference/androidx/core/content/FileProvider).
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
