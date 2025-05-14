# Types containing unmanaged code
Microsoft defines two broad categories for source code. Managed code compiles into bytecode and is then executed by a virtual machine. Unmanaged code is compiled directly into machine code. All C\# code is managed but it is possible to call external unmanaged code. This metric counts the number of `extern` methods in each class that are implemented by unmanaged code. Managed code has many advantages over unmanaged code such as built in memory management performed by the virtual machine and the ability to run compiled programs on a wider variety of architectures.


## Recommendation
Consider whether the unmanaged methods could be replaced by managed code instead.


## Example
This example shows a function that displays a message box when clicked. It is implemented with unmanaged code from the `User32.dll` library.


```csharp
using System;
using System.Windows.Forms;
using System.Runtime.InteropServices;

public partial class UnmanagedCodeExample : Form
{
    [DllImport("User32.dll")]
    public static extern int MessageBox(int h, string m, string c, int type); // violation

    private void btnSayHello_Click(object sender, EventArgs e)
    {
        MessageBox(0, "Hello World", "Title", 0);
    }
}

```

## Fixing by Using Managed Code
This example uses managed code to perform the same function.


```csharp
using System;
using System.Windows.Forms;

public partial class ManagedCodeExample : Form
{
    private void btnSayHello_Click(object sender, EventArgs e)
    {
         MessageBox.Show("Hello World", "Title");
    }
}

```

## References
* MSDN, C\# Reference. [extern](http://msdn.microsoft.com/en-us/library/e59b22c5(v=vs.80).aspx).
* Wikipedia. [Managed code](http://en.wikipedia.org/wiki/Managed_code).
