// Copyright 2026 Google LLC
//
// Exercises the installed Mozc server with a fixed, non-sensitive context.

#include <iostream>

#include "client/client.h"
#include "protocol/commands.pb.h"

int main() {
  mozc::client::Client client;
  if (!client.EnsureSession()) {
    std::cerr << "Could not connect to Mozc server\n";
    return 1;
  }

  mozc::commands::Context context;
  context.set_preceding_text("今日は晴れです。\n明日は");
  mozc::commands::Output output;
  mozc::commands::KeyEvent key;
  key.set_special_key(mozc::commands::KeyEvent::ON);
  if (!client.SendKeyWithContext(key, context, &output)) {
    std::cerr << "Could not enable IME\n";
    return 1;
  }
  for (const char letter : {'t', 'e', 'n', 'k', 'i'}) {
    key.Clear();
    key.set_key_code(letter);
    if (!client.SendKeyWithContext(key, context, &output)) {
      std::cerr << "Could not send key to Mozc server\n";
      return 1;
    }
  }

  key.Clear();
  key.set_special_key(mozc::commands::KeyEvent::SPACE);
  if (!client.SendKeyWithContext(key, context, &output)) {
    std::cerr << "Could not request conversion\n";
    return 1;
  }
  if (!output.has_candidate_window() &&
      !client.SendKeyWithContext(key, context, &output)) {
    std::cerr << "Could not open candidate window\n";
    return 1;
  }
  std::cout << "Mozc conversion: candidate window="
            << output.has_candidate_window()
            << ", preedit=" << output.has_preedit() << '\n';
  return output.has_candidate_window() ? 0 : 1;
}
