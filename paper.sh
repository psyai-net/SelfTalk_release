#!/usr/bin/env bash

user_agent="Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/111.0"

arxiv_download() {
  id="$1"
  name="$2"
  year="${3:-2023}"
  mkdir -p "paper/${year}-${name}"
  wget -c --user-agent="${user_agent}" "https://arxiv.org/pdf/${id}" -O "paper/${year}-${name}.pdf"
  wget -c --user-agent="${user_agent}" "https://arxiv.org/e-print/${id}" -O "paper/${year}-${name}/${id}"

  tar -xf "paper/${year}-${name}/${id}" -C "paper/${year}-${name}/" && rm "paper/${year}-${name}/${id}"
}

arxiv_download "2306.10799" "selftalk" 2023

