PYTHON ?= python

.PHONY: setup audit-data test lint train evaluate validate validate-final notebook-final package package-final run-app build-docker

setup:
	$(PYTHON) -m pip install -r requirements-dev.txt

audit-data:
	$(PYTHON) scripts/run_full_pipeline.py --audit-only

test:
	$(PYTHON) -m compileall -q src app_components streamlit_app.py scripts tests
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) scripts/check_notebook.py

lint:
	ruff check src app_components streamlit_app.py scripts tests

train:
	$(PYTHON) scripts/run_full_pipeline.py --retrain

evaluate:
	$(PYTHON) scripts/run_full_pipeline.py --skip-train

validate:
	$(PYTHON) scripts/validate_project.py

validate-final:
	$(PYTHON) scripts/validate_project.py --require-final

notebook-final:
	$(PYTHON) scripts/check_notebook.py --require-executed --require-full-run

package:
	$(PYTHON) scripts/build_submission_manifest.py
	$(PYTHON) scripts/package_submission.py
	$(PYTHON) scripts/verify_submission_package.py ../TUNGDUONG_flower-image-restoration-cnn_95plus_merged.zip

package-final:
	$(PYTHON) scripts/build_submission_manifest.py
	$(PYTHON) scripts/package_submission.py --final

run-app:
	streamlit run streamlit_app.py

build-docker:
	docker build -t flower-restoration-streamlit .
