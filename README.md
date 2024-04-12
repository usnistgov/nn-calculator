# NN-Calculator GitHub pages
Neural Network Calculator

# Deployment instructions

- copy the files from the master branch, dist folder (after npm run build) to the dist folder in nist-pages branch

- copy the index.html from dist to _include folder (make sure that the path to all loaded .js and .css files is correct)

- update the time stamp of index.html in the root folder  in order to trigger the deployment.

- commit the changes to nist-pages branch

# Current configuration of the nist-pages

1. Web app at https://pages.nist.gov/nn-calculator/.
    * The web app is deployed using the Pages configuration: https://github.com/usnistgov/nn-calculator/settings/pages.
    * It is deploying the web app from a branch using the /root/_include folder and the index.html file. 
    * The index.html file must load .js and .css files from a specified directory which is /root//dist in my case.
2. Documentations at https://usnistgov.github.io/nn-calculator/docs/about.html
    * The documentation is deployed using Jekyll which has a configuration file in /root/_config.yaml.
    * The configuration file points to the /root/docs where one has to place all documentation .html files.

- Note 1: See the wiki at https://github.com/usnistgov/pages-root/wiki for instructions on configuring repositories for publishing.
- Note 2: For inspiration, see the repo https://github.com/usnistgov/eerc and look at the script named “publish-on-nist-pages” 

 

