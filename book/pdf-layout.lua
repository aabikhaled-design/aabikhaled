function Code(inline)
  if FORMAT ~= "latex" then
    return nil
  end
  local escaped = inline.text:gsub("([\\%%#{}%^ &$~_])", "\\%1")
  return pandoc.RawInline("latex", "\\EscVerb{" .. escaped .. "}")
end

function Table(block)
  if FORMAT ~= "latex" then
    return nil
  end
  return block:walk({
    Str = function(inline)
      if #inline.text <= 20 or not inline.text:find("[/_=(]") then
        return nil
      end
      local wrapped = pandoc.List()
      for _, codepoint in utf8.codes(inline.text) do
        if #wrapped > 0 then
          wrapped:insert(pandoc.RawInline("latex", "\\allowbreak{}"))
        end
        wrapped:insert(pandoc.Str(utf8.char(codepoint)))
      end
      return wrapped
    end,
  })
end
