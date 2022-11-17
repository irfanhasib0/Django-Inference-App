npm run build
cp -r ./build/* ../../irfanhasib0.github.io/docs/
cd ../../irfanhasib0.github.io/docs/ && git add ../docs
cd ../../irfanhasib0.github.io/docs/ && git commit -m "deploying new version"
cd ../../irfanhasib0.github.io/docs/ && git push origin master
