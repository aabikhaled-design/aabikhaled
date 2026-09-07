function Table(block)
  if FORMAT ~= "latex" then
    return nil
  end
  return block:walk({
    Code = function(inline)
      local escaped = inline.text:gsub("([\\%%#{}%^ &$~_])", "\\%1")
      return pandoc.RawInline("latex", "\\EscVerb{" .. escaped .. "}")
    end,
  })
end
