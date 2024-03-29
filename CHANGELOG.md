# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [v0.2.5](https://github.com/epwalsh/batched-fn/releases/tag/v0.2.5) - 2024-03-10

### Fixed

- Removed unnecessary `Mutex` in `BatchedFn`.
- Fixed issue where thread would crash when request from calling thread disconnects.

## [v0.2.4](https://github.com/epwalsh/batched-fn/releases/tag/v0.2.4) - 2022-03-14

### Added

- Added more documentation about `channel_cap`.

## [v0.2.3](https://github.com/epwalsh/batched-fn/releases/tag/v0.2.3) - 2022-03-11

### Changed

- Minor updates to documentation and CI.
