#
# Cargo helpers.
#

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

#
# Git helpers.
#

.PHONY: create-branch
create-branch :
ifneq ($(issue),)
	git checkout -b ISSUE-$(issue)
	git push --set-upstream origin $$(git branch | grep \* | cut -d ' ' -f2)
else ifneq ($(name),)
	git checkout -b $(name)
	git push --set-upstream origin $$(git branch | grep \* | cut -d ' ' -f2)
else
	$(error must supply 'issue' or 'name' parameter)
endif

.PHONY : delete-branch
delete-branch :
	@BRANCH=`git rev-parse --abbrev-ref HEAD` \
		&& [ $$BRANCH != 'master' ] \
		&& echo "On branch $$BRANCH" \
		&& echo "Checking out master" \
		&& git checkout master \
		&& git pull \
		&& echo "Deleting branch $$BRANCH" \
		&& git branch -d $$BRANCH \
		&& git remote prune origin
