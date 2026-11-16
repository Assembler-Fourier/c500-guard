// BarrierWatcher's own Windows window (15 Nov 2026). It replaces the Chrome app window: own icon, own taskbar
// button, no browser menus or tabs. It is only the window; starting and stopping the alarm system stays with the
// launcher (barrierwatcher_app.pyw), which runs this exe and follows its process.
// Built with the C# compiler that ships with .NET Framework 4.8 (C# 5) and Microsoft's WebView2 SDK.
using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace BarrierWatcherApp
{
    class MainWindow : Form
    {
        readonly WebView2 view = new WebView2();
        readonly bool confirmClose;

        public MainWindow(string url, string dataFolder, bool confirm, string browserArgs)
        {
            confirmClose = confirm;
            Text = "BarrierWatcher";
            BackColor = Color.FromArgb(8, 9, 8);
            StartPosition = FormStartPosition.CenterScreen;
            Size = new Size(1280, 800);
            MinimumSize = new Size(640, 420);
            WindowState = FormWindowState.Maximized;
            try { Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath); } catch (Exception) { }
            view.Dock = DockStyle.Fill;
            view.DefaultBackgroundColor = BackColor;
            view.CreationProperties = new CoreWebView2CreationProperties();
            view.CreationProperties.UserDataFolder = dataFolder;
            // Lean engine settings: the PC is short of memory, and this window only shows one local page.
            view.CreationProperties.AdditionalBrowserArguments = browserArgs;
            Controls.Add(view);
            view.CoreWebView2InitializationCompleted += OnReady;
            view.Source = new Uri(url);
            FormClosing += OnClosing;
        }

        void OnReady(object sender, CoreWebView2InitializationCompletedEventArgs e)
        {
            if (!e.IsSuccess)
            {
                MessageBox.Show(this, "The window engine (Microsoft WebView2) could not start:\n" +
                                e.InitializationException.Message, "BarrierWatcher",
                                MessageBoxButtons.OK, MessageBoxIcon.Error);
                Environment.Exit(3);
                return;
            }
            CoreWebView2Settings settings = view.CoreWebView2.Settings;
            settings.AreDevToolsEnabled = false;
            settings.IsStatusBarEnabled = false;
            settings.AreBrowserAcceleratorKeysEnabled = true;     // F5 still reloads
            settings.IsZoomControlEnabled = true;
            view.CoreWebView2.DocumentTitleChanged += delegate
            {
                string title = view.CoreWebView2.DocumentTitle;
                if (!string.IsNullOrEmpty(title) && !title.StartsWith("127.0.0.1")) Text = title;
            };
            view.CoreWebView2.NewWindowRequested += OnNewWindow;
            Resize += delegate
            {
                // Minimised: let the engine give memory back.
                view.CoreWebView2.MemoryUsageTargetLevel = WindowState == FormWindowState.Minimized
                    ? CoreWebView2MemoryUsageTargetLevel.Low : CoreWebView2MemoryUsageTargetLevel.Normal;
            };
        }

        void OnNewWindow(object sender, CoreWebView2NewWindowRequestedEventArgs e)
        {
            // A page asking for a new window opens in a second BarrierWatcher window, never in a web browser.
            CoreWebView2Deferral deferral = e.GetDeferral();
            PopupWindow popup = new PopupWindow(Icon, BackColor);
            popup.Show(this);
            popup.Attach(view.CoreWebView2.Environment, e, deferral);
        }

        void OnClosing(object sender, FormClosingEventArgs e)
        {
            // Only a person closing the window is asked; Windows shutting down or the launcher ending it is not.
            if (!confirmClose || e.CloseReason != CloseReason.UserClosing) return;
            DialogResult answer = MessageBox.Show(this,
                "Closing BarrierWatcher switches the whole alarm system off.\n\nClose it?", "BarrierWatcher",
                MessageBoxButtons.OKCancel, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2);
            if (answer != DialogResult.OK) e.Cancel = true;
        }
    }

    class PopupWindow : Form
    {
        readonly WebView2 view = new WebView2();

        public PopupWindow(Icon icon, Color back)
        {
            Text = "BarrierWatcher";
            Icon = icon;
            BackColor = back;
            Size = new Size(1100, 800);
            StartPosition = FormStartPosition.CenterParent;
            view.Dock = DockStyle.Fill;
            view.DefaultBackgroundColor = back;
            Controls.Add(view);
        }

        public async void Attach(CoreWebView2Environment environment, CoreWebView2NewWindowRequestedEventArgs e,
                                 CoreWebView2Deferral deferral)
        {
            try
            {
                await view.EnsureCoreWebView2Async(environment);
                view.CoreWebView2.Settings.AreDevToolsEnabled = false;
                view.CoreWebView2.DocumentTitleChanged += delegate { Text = view.CoreWebView2.DocumentTitle; };
                e.NewWindow = view.CoreWebView2;
                e.Handled = true;
            }
            catch (Exception)
            {
                Close();
            }
            finally
            {
                deferral.Complete();
            }
        }
    }

    static class Program
    {
        [STAThread]
        static int Main(string[] args)
        {
            string url = "http://127.0.0.1:8787/?app=1";
            string data = Path.Combine(Path.Combine(Environment.GetFolderPath(
                Environment.SpecialFolder.LocalApplicationData), "c500-guard"), "app-window-webview");
            bool confirm = true;
            string browserArgs = "--disable-gpu --renderer-process-limit=1 --disable-background-networking " +
                                 "--disable-component-update --disable-extensions --disable-sync " +
                                 "--disable-features=msSmartScreenProtection,msWebOOUI,msPdfOOUI,Translate";
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--url" && i + 1 < args.Length) url = args[++i];
                else if (args[i] == "--data" && i + 1 < args.Length) data = args[++i];
                else if (args[i] == "--no-confirm") confirm = false;
                else if (args[i] == "--browser-args" && i + 1 < args.Length) browserArgs = args[++i];
            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainWindow(url, data, confirm, browserArgs));
            return 0;
        }
    }
}
