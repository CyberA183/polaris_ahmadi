# Building Polaris with Briefcase

This project uses Briefcase for packaging:

- Desktop app: `polaris_desktop` (macOS, Windows)
- Mobile app: `polaris_mobile` (Android, iOS)

## 1) Base prerequisites (all platforms)

- Python 3.12
- `pip install -r requirements.txt`
- `pip install briefcase`

Optional local helper scripts:

- Bash: `scripts/briefcase_build.sh`
- PowerShell: `scripts/briefcase_build.ps1`

## 2) macOS desktop build

Prerequisites:

- macOS host
- Xcode Command Line Tools (`xcode-select --install`)

Commands:

```bash
./scripts/briefcase_build.sh macOS polaris_desktop all
```

Equivalent manual commands:

```bash
briefcase create macOS -a polaris_desktop --no-input
briefcase build macOS -a polaris_desktop --no-input
briefcase package macOS -a polaris_desktop --no-input
```

## 3) Windows desktop build

Prerequisites:

- Windows host
- Visual Studio Build Tools (C++ workload)

Commands (PowerShell):

```powershell
.\scripts\briefcase_build.ps1 -Target windows -App polaris_desktop -Step all
```

Equivalent manual commands:

```powershell
briefcase create windows -a polaris_desktop --no-input
briefcase build windows -a polaris_desktop --no-input
briefcase package windows -a polaris_desktop --no-input
```

## 4) Android build

Prerequisites:

- macOS/Linux/Windows host
- Java 17 (Temurin recommended)
- Android SDK + command-line tools
- Android SDK env vars:
  - `ANDROID_HOME` (or `ANDROID_SDK_ROOT`)
  - `JAVA_HOME`

Commands:

```bash
./scripts/briefcase_build.sh android polaris_mobile all
```

Equivalent manual commands:

```bash
briefcase create android -a polaris_mobile --no-input
briefcase build android -a polaris_mobile --no-input
briefcase package android -a polaris_mobile --no-input
```

## 5) iOS build

Prerequisites:

- macOS host
- Xcode (full app, not only CLT)
- iOS simulator runtimes installed in Xcode

Commands:

```bash
./scripts/briefcase_build.sh iOS polaris_mobile all
```

Equivalent manual commands:

```bash
briefcase create iOS -a polaris_mobile --no-input
briefcase build iOS -a polaris_mobile --no-input
briefcase package iOS -a polaris_mobile --no-input
```

## 6) Running single steps

Bash:

```bash
./scripts/briefcase_build.sh macOS polaris_desktop create
./scripts/briefcase_build.sh macOS polaris_desktop build
./scripts/briefcase_build.sh macOS polaris_desktop package
```

PowerShell:

```powershell
.\scripts\briefcase_build.ps1 -Target windows -App polaris_desktop -Step create
.\scripts\briefcase_build.ps1 -Target windows -App polaris_desktop -Step build
.\scripts\briefcase_build.ps1 -Target windows -App polaris_desktop -Step package
```
