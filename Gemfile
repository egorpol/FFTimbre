source "https://rubygems.org"

# Build with the GitHub Pages gemset on GitHub's infrastructure
gem "github-pages", group: :jekyll_plugins

# Local-only helpers can be added in a dev group without affecting Pages
group :development do
  # Ruby 3+ needs WEBrick for `jekyll serve` locally
  gem "webrick", "~> 1.8"
  # Optional: faster file watching on Windows
  # gem "wdm", "~> 0.1.1", platforms: [:mingw, :x64_mingw, :mswin]
end
