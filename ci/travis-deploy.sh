#!/bin/bash
echo "Running deployment script..."
conda install --yes jupyter
pip install pdoc==0.3.2 pygments

# Generating documentation
cd ~
mkdir -p ./doc/carl/notebooks
cd ./doc/carl/notebooks

OIFS="$IFS"
IFS=$'\n'
for nb in ${TRAVIS_BUILD_DIR}/examples/*ipynb; do
    jupyter nbconvert "$nb" --to markdown
done
cp ${TRAVIS_BUILD_DIR}/examples/*md .
cp -r ${TRAVIS_BUILD_DIR}/examples/*_files .
IFS="$OIFS"

cd ~
python ${TRAVIS_BUILD_DIR}/ci/make_doc.py --overwrite --html --html-dir ./doc --template-dir ${TRAVIS_BUILD_DIR}/ci/templates --notebook-dir ./doc/carl/notebooks carl

# Copying to github pages
echo "Copying built files"
git clone -b gh-pages "https://${GH_TOKEN}@github.com/diana-hep/carl.git" deploy > /dev/null 2>&1 || exit 1
cd deploy
git rm -r notebooks/*
cd ..
cp -r ./doc/carl/* deploy

# Move into deployment directory
cd deploy

# Commit changes, allowing empty changes (when unchanged)
echo "Committing and pushing to Github"
git config user.name "Travis-CI"
git config user.email "travis@yoursite.com"
git add -A
git commit --allow-empty -m "Deploying documentation for ${TRAVIS_COMMIT}" || exit 1

# Push to branch
git push origin gh-pages > /dev/null 2>&1 || exit 1

echo "Pushed deployment successfully"
exit 0
