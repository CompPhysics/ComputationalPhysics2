#!/bin/sh
set -x

function system {
  "$@"
  if [ $? -ne 0 ]; then
    echo "make.sh: unsuccessful command $@"
    echo "abort!"
    exit 1
  fi
}

if [ $# -eq 0 ]; then
echo 'bash make.sh slides1|slides2'
exit 1
fi

name=$1
rm -f *.tar.gz

opt="--encoding=utf-8"
# Note: Makefile examples contain constructions like ${PROG} which
# looks like Mako constructions, but they are not. Use --no_mako
# to turn off Mako processing.
opt="--no_mako"

rm -f *.aux




# IPython notebook
system doconce format ipynb $name $opt  --no_abort



# Ordinary plain LaTeX document
rm -f *.aux  # important after beamer
system doconce format pdflatex $name --minted_latex_style=trac --latex_admon=paragraph $opt
system doconce ptex2tex $name envir=minted
# Add special packages
doconce subst "% Add user's preamble" "\g<1>\n\\usepackage{simplewick}" $name.tex
doconce replace 'section{' 'section*{' $name.tex
pdflatex -shell-escape $name
pdflatex -shell-escape $name
mv -f $name.pdf ${name}.pdf
cp $name.tex ${name}-plain.tex



# Publish
dest=../../pub
if [ ! -d $dest/$name ]; then
mkdir $dest/$name
mkdir $dest/$name/pdf
mkdir $dest/$name/html
mkdir $dest/$name/ipynb
fi
cp ${name}*.pdf $dest/$name/pdf
cp -r ${name}*.html ._${name}*.html reveal.js $dest/$name/html

# Figures: cannot just copy link, need to physically copy the files
if [ -d fig-${name} ]; then
if [ ! -d $dest/$name/html/fig-$name ]; then
mkdir $dest/$name/html/fig-$name
fi
cp -r fig-${name}/* $dest/$name/html/fig-$name
fi

cp ${name}.ipynb $dest/$name/ipynb
ipynb_tarfile=ipynb-${name}-src.tar.gz
if [ ! -f ${ipynb_tarfile} ]; then
cat > README.txt <<EOF
This IPython notebook ${name}.ipynb does not require any additional
programs.
EOF
tar czf ${ipynb_tarfile} README.txt
fi
cp ${ipynb_tarfile} $dest/$name/ipynb
