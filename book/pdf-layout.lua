function Code(inline)
  if FORMAT ~= "latex" then
    return nil
  end
  local escaped = inline.text:gsub("([\\%%#{}%^ &$~_])", "\\%1")
  return pandoc.RawInline("latex", "\\EscVerb{" .. escaped .. "}")
end
