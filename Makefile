.PHONY : build
build :
	cargo build

.PHONY : format
format :
	cargo fmt --


.PHONY : lint
lint :
	cargo fmt --all -- --check
	cargo clippy --all-targets -- \
			-D warnings \
			-A clippy::let_and_return \
			-A clippy::redundant_clone

.PHONY : test
test :
	@cargo test --lib
	@cargo test --doc
	@cargo test --test codegen
	@cargo test --test runtime

.PHONY : doc
doc :
	cargo doc

.PHONY : readme
readme :
	cargo readme > README.md
