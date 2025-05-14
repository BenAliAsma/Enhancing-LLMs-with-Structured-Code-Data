# Insecure loading of an Android Dex File
It is dangerous to load Dex libraries from shared world-writable storage spaces. A malicious actor can replace a dex file with a maliciously crafted file which when loaded by the app can lead to code execution.


## Recommendation
Loading a file from private storage instead of a world-writable one can prevent this issue, because the attacker cannot access files stored there.


## Example
The following example loads a Dex file from a shared world-writable location. in this case, since the \`/sdcard\` directory is on external storage, anyone can read/write to the location. bypassing all Android security policies. Hence, this is insecure.


```java

import android.app.Application;
import android.content.Context;
import android.content.pm.PackageInfo;
import android.os.Bundle;

import dalvik.system.DexClassLoader;
import dalvik.system.DexFile;

public class InsecureDexLoading extends Application {
	@Override
	public void onCreate() {
		super.onCreate();
		updateChecker();
	}

	private void updateChecker() {
		try {
			File file = new File("/sdcard/updater.apk");
			if (file.exists() && file.isFile() && file.length() <= 1000) {
				DexClassLoader cl = new DexClassLoader(file.getAbsolutePath(), getCacheDir().getAbsolutePath(), null,
						getClassLoader());
				int version = (int) cl.loadClass("my.package.class").getDeclaredMethod("myMethod").invoke(null);
				if (Build.VERSION.SDK_INT < version) {
					Toast.makeText(this, "Loaded Dex!", Toast.LENGTH_LONG).show();
				}
			}
		} catch (Exception e) {
			// ignore
		}
	}
}

```
The next example loads a Dex file stored inside the app's private storage. This is not exploitable as nobody else except the app can access the data stored there.


```java
public class SecureDexLoading extends Application {
	@Override
	public void onCreate() {
		super.onCreate();
		updateChecker();
	}

	private void updateChecker() {
		try {
			File file = new File(getCacheDir() + "/updater.apk");
			if (file.exists() && file.isFile() && file.length() <= 1000) {
				DexClassLoader cl = new DexClassLoader(file.getAbsolutePath(), getCacheDir().getAbsolutePath(), null,
						getClassLoader());
				int version = (int) cl.loadClass("my.package.class").getDeclaredMethod("myMethod").invoke(null);
				if (Build.VERSION.SDK_INT < version) {
					Toast.makeText(this, "Securely loaded Dex!", Toast.LENGTH_LONG).show();
				}
			}
		} catch (Exception e) {
			// ignore
		}
	}
}
```

## References
* Android Documentation: [Data and file storage overview](https://developer.android.com/training/data-storage/).
* Android Documentation: [DexClassLoader](https://developer.android.com/reference/dalvik/system/DexClassLoader).
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
