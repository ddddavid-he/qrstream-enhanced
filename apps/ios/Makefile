# Makefile for the iOS app — wraps the long xcodebuild/devicectl invocations.
#
# Usage:
#   make bootstrap                         # one-time setup: fetch SwiftPM checkouts
#   make generate                          # regenerate QRStreamApp.xcodeproj
#   make build-sim                         # build for the iOS Simulator
#   make build-device DEVICE=<UDID>        # build for a real device
#   make install DEVICE=<UDID>             # install last-built device .app
#   make run DEVICE=<UDID>                 # install + launch
#   make devices                           # list connected real devices
#   make clean                             # wipe DerivedData + SourcePackages

# -- Configuration ------------------------------------------------------------

# Use Xcode.app rather than CommandLineTools (the macOS default).
DEVELOPER_DIR ?= /Applications/Xcode.app/Contents/Developer
export DEVELOPER_DIR

# Override at the command line if you have multiple Apple IDs / teams.
DEVELOPMENT_TEAM ?=
export DEVELOPMENT_TEAM

PROJECT      := QRStreamApp.xcodeproj
SCHEME       := QRStreamApp
BUNDLE_ID    := dev.qrstream.app
SOURCE_PKGS  := SourcePackages

# Pin to a specific simulator if set; otherwise xcodebuild picks one matching
# `name=iPhone 17 Pro`.
SIM_NAME     ?= iPhone 17 Pro

# Where xcodebuild puts the built .app for each platform.
DERIVED_ROOT := $(HOME)/Library/Developer/Xcode/DerivedData

# -- Helpers ------------------------------------------------------------------

XCODEBUILD_FLAGS := \
	-project $(PROJECT) \
	-scheme $(SCHEME) \
	-configuration Debug \
	-clonedSourcePackagesDirPath $(SOURCE_PKGS)

# -- Targets ------------------------------------------------------------------

.PHONY: bootstrap generate build-sim build-device install run devices clean help

help:
	@awk 'BEGIN { FS = ":.*?##" } /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

bootstrap: $(SOURCE_PKGS)/checkouts/zxing-cpp/zint/CMakeLists.txt ## Fetch SwiftPM dependencies (zxing-cpp + zint submodule)

# SwiftPM's `xcodebuild -resolvePackageDependencies` occasionally fails to
# initialize the zxing-cpp `zint` submodule because its shallow-clone metadata
# is created in a directory that doesn't exist yet. Doing the clone manually
# sidesteps that race entirely. Subsequent `xcodebuild` runs are happy because
# they only check that the checkout exists, not how it got there.
$(SOURCE_PKGS)/checkouts/zxing-cpp/zint/CMakeLists.txt:
	@mkdir -p $(SOURCE_PKGS)/checkouts
	@if [ ! -d $(SOURCE_PKGS)/checkouts/zxing-cpp ]; then \
		echo "==> Cloning zxing-cpp (this may take a minute)"; \
		git clone --depth 1 --branch v3.0.2 \
			https://github.com/zxing-cpp/zxing-cpp \
			$(SOURCE_PKGS)/checkouts/zxing-cpp; \
	fi
	@echo "==> Initializing zint submodule"
	@cd $(SOURCE_PKGS)/checkouts/zxing-cpp && \
		git submodule update --init --recursive --depth 1

generate: bootstrap ## Regenerate the .xcodeproj from project.yml
	@if [ -z "$(DEVELOPMENT_TEAM)" ]; then \
		echo "warning: DEVELOPMENT_TEAM not set — device builds will fail to sign."; \
		echo "         Set it via 'make generate DEVELOPMENT_TEAM=<team-id>' for device builds."; \
	fi
	xcodegen generate

build-sim: generate ## Build for the iOS Simulator (no signing)
	xcodebuild $(XCODEBUILD_FLAGS) \
		-destination "platform=iOS Simulator,name=$(SIM_NAME)" \
		CODE_SIGNING_ALLOWED=NO \
		build

build-device: generate ## Build for a connected device (requires DEVICE=<udid>)
	@if [ -z "$(DEVICE)" ]; then echo "error: DEVICE=<udid> is required (run 'make devices')"; exit 1; fi
	xcodebuild $(XCODEBUILD_FLAGS) \
		-destination "id=$(DEVICE)" \
		-allowProvisioningUpdates \
		build

install: ## Install the last-built device .app onto DEVICE
	@if [ -z "$(DEVICE)" ]; then echo "error: DEVICE=<udid> is required"; exit 1; fi
	@APP_PATH=$$(find $(DERIVED_ROOT)/QRStreamApp-*/Build/Products/Debug-iphoneos -maxdepth 2 -name "QRStreamApp.app" 2>/dev/null | head -1); \
	if [ -z "$$APP_PATH" ]; then echo "error: no built .app found — run 'make build-device' first"; exit 1; fi; \
	echo "==> Installing $$APP_PATH"; \
	xcrun devicectl device install app --device $(DEVICE) "$$APP_PATH"

run: build-device install ## Build, install, and launch on DEVICE
	xcrun devicectl device process launch --device $(DEVICE) $(BUNDLE_ID)

devices: ## List connected iOS devices and their UDIDs
	xcrun devicectl list devices

clean: ## Wipe DerivedData and SwiftPM checkouts
	rm -rf $(DERIVED_ROOT)/QRStreamApp-*
	rm -rf $(SOURCE_PKGS)
	rm -rf $(PROJECT)
