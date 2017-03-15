git add -u .

echo 'Enter the commit message:'
read commitMessage

git commit -m "$commitMessage"

git push origin master