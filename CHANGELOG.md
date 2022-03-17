# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- ⚠️ Breaking change ⚠️

  The `batched_fn!` macro now returns a `BatchedFn` instance instead of a closure, which allows
  us to avoid requiring static parameters to the macro.
  It's a simple change to upgrade your codebase. If your code looked like this before:

  ```rust
  let batch_predictor = batched_fn! { ... };
  return batch_predictor(input).await;
  ```

  You just need to change the last line to:

  ```rust
  return batch_predictor.evaluate_in_batch(input).await;
  ```

## [v0.2.4](https://github.com/epwalsh/batched-fn/releases/tag/v0.2.4) - 2022-03-14

### Added

- Added more documentation about `channel_cap`.

## [v0.2.3](https://github.com/epwalsh/batched-fn/releases/tag/v0.2.3) - 2022-03-11

### Changed

- Minor updates to documentation and CI.
